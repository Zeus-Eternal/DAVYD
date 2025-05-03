#!/usr/bin/env python3
# src/davyd.py

import json
import re
from json import JSONDecodeError
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from model_providers import BaseModelClient, GenerationConfig
from utils.manage_dataset import DatasetManager
from prompt_engineering import PromptEngineer
from autogen_client.template_manager import TemplateManager
from autogen_client.data_validator import DataValidator
from autogen_client.visualization import Visualization
from autogen_client.assistant_agent import AssistantAgent
from utils.proxy_agent import ProxyAgent

logger = logging.getLogger(__name__)


class Davyd:
    """
    Core class for generating, validating, and saving synthetic datasets.
    Supports both direct LLM generation and an AutogenAgent fallback.
    """

    def __init__(
        self,
        num_entries: int,
        model_client: BaseModelClient,
        model_name: str,
        dataset_manager: DatasetManager,
        section_separator: str = "|",
        data_separator: str = '"',
        quality_level: int = 2,  # 1=Fast, 2=Balanced, 3=High
    ):
        self.num_entries       = num_entries
        self.model_client      = model_client
        self.model_name        = model_name
        self.dataset_manager   = dataset_manager
        self.section_separator = section_separator
        self.data_separator    = data_separator

        # Helper components
        self.prompt_engineer  = PromptEngineer()
        self.template_manager = TemplateManager()
        self.data_validator    = DataValidator()
        self.visualization     = Visualization()

        # Setup assistant‐based fallback
        self.set_quality(quality_level)
        agent_cfg = {
            "model_client": self.model_client,
            "model_name":   self.model_name,
            "max_retries":  getattr(self, "max_retries", 1)
        }
        self.assistant_agent = AssistantAgent(agent_cfg, "assistant")
        proxy_cfg = {**agent_cfg, "num_agents": 1}
        self.proxy_agent     = ProxyAgent(proxy_cfg)

        self.dataset = pd.DataFrame()
        self.fields  = []
        logger.info(
            f"Davyd initialized: entries={num_entries}, "
            f"model={model_name}, sep='{section_separator}', "
            f"wrap='{data_separator}', quality={quality_level}"
        )

    def set_quality(self, level: int):
        """Configure generation parameters based on quality level."""
        presets = {
            1: dict(temperature=0.7, top_p=1.0, max_retries=1),
            2: dict(temperature=0.5, top_p=0.9, max_retries=2),
            3: dict(temperature=0.3, top_p=0.7, max_retries=3),
        }
        cfg = presets.get(level, presets[2])
        self.temperature = cfg["temperature"]
        self.top_p       = cfg["top_p"]
        self.max_retries = cfg["max_retries"]

    def generate_dataset(
        self,
        heading: str,
        example_rows: List[str]
    ) -> pd.DataFrame:
        """
        Generate dataset using AutogenAgent as the sole generation method.
        Retries on failure and returns partial if ≥50% succeeded.
        """
        if not self._validate_heading_format(heading):
            raise ValueError("Invalid heading format")

        self.fields = [
            f.strip().strip(self.data_separator)
            for f in heading.split(self.section_separator)
        ]

        remaining = self.num_entries
        records: List[Dict[str, Any]] = []

        while remaining > 0:
            prompt = self._build_prompt(remaining, heading, example_rows)
            logger.debug(f"[Autogen Prompt]\n{prompt}")

            try:
                raw = self._autogen_generate(prompt)
                new_recs = self._parse_response(raw)
                valid = [r for r in new_recs if self._is_valid_entry(r)]
                records.extend(valid)
                remaining = self.num_entries - len(records)
                logger.info(f"✅ Autogen: {len(valid)} valid records | {remaining} remaining")

            except Exception as e:
                logger.exception("❌ Autogen generation failed")
                if len(records) >= self.num_entries * 0.5:
                    break
                raise

        self.dataset = pd.DataFrame(records[: self.num_entries])
        if len(self.dataset) < self.num_entries:
            logger.warning(
                f"⚠️ Partial dataset generated: {len(self.dataset)}/{self.num_entries} entries"
            )
        return self.dataset

    def _build_prompt(
        self,
        remaining: int,
        heading: str,
        example_rows: List[str]
    ) -> str:
        """Construct prompt with error‐resilient templating."""
        try:
            tmpl = self.template_manager.load_templates().get("generation", {})
            if isinstance(tmpl, dict) and "prompt" in tmpl:
                return tmpl["prompt"].format(
                    fields=heading,
                    examples="\n".join(example_rows),
                    remaining=remaining
                )
        except Exception:
            logger.warning("Autogen template missing; falling back")

        # Standard PromptEngineer
        return self.prompt_engineer.build_generation_prompt(
            fields=[f.strip() for f in heading.split(self.section_separator) if f.strip()],
            examples=example_rows,
            num_entries=remaining
        )

    def _autogen_generate(self, prompt: str) -> str:
        """Use ProxyAgent+AssistantAgent to get a string reply."""
        self.proxy_agent.initiate_chat(self.assistant_agent, message=prompt)
        raw = self.assistant_agent.last_message()
        if not isinstance(raw, str):
            logger.warning(f"Non‐string autogen response: {raw!r}")
            return ""
        logger.debug(f"[AUTOGEN RAW RESPONSE]\n{raw}")
        return raw

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        """Parse JSON list/dict first, else line‐by‐line via separator."""
        logger.debug(f"Raw output:\n{raw}")
        # JSON‐first
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except JSONDecodeError:
            pass

        # Fallback: split lines by section_separator
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith(("```", "#", "---")):
                continue
            parts = [p.strip() for p in line.split(self.section_separator)]
            if len(parts) == len(self.fields):
                records.append(dict(zip(self.fields, parts)))
        return records

    def _is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """Ensure no field is missing or empty."""
        return all(entry.get(f) not in (None, "") for f in self.fields)

    def _validate_heading_format(self, heading: str) -> bool:
        """Heading must wrap fields in data_separator and use section_separator."""
        parts = heading.split(self.section_separator)
        return (
            len(parts) > 1
            and all(p.startswith(self.data_separator) and p.endswith(self.data_separator) for p in parts)
        )

    def save_dataset(self, base_name: str = "dataset") -> str:
        """Save current DataFrame to a temp CSV and return its path."""
        if self.dataset.empty:
            raise ValueError("No dataset to save")
        filepath = self.dataset_manager.get_temp_filename(base_name)
        self.dataset_manager.save_dataset(self.dataset, filepath, which="temp", overwrite=False)
        logger.info("Dataset saved to %s", filepath)
        return filepath

    def visualize(self, **kwargs) -> Any:
        """Delegate rendering to Visualization."""
        return self.visualization.render(self.dataset, **kwargs)

    def validate_entry(self, entry: str) -> str:
        """LLM‐driven single‐row validation."""
        prompt = self.prompt_engineer.build_validation_prompt(entry, self.fields)
        return self.model_client.generate_text(prompt, temperature=0.1)
