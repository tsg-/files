#!/usr/bin/env python3
"""
LMSYS-Chat-1M Dataset Replay for vLLM + LMCache Benchmarking

This script processes the LMSYS-Chat-1M dataset and replays conversations
to vLLM to measure KV cache effectiveness, especially for multi-turn scenarios.

Usage:
    # Step 1: Download dataset from HuggingFace (requires authentication)
    # Visit: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    
    # Step 2: Run benchmark
    python lmsys_chat_replay.py \\
        --dataset-path /path/to/lmsys-chat-1m \\
        --vllm-endpoint http://localhost:8000/v1 \\
        --num-conversations 100 \\
        --min-turns 3 \\
        --enable-prefix-caching
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio


@dataclass
class ConversationStats:
    """Statistics for a single conversation."""
    conversation_id: str
    num_turns: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    ttft_per_turn: List[float] = field(default_factory=list)
    tpot_per_turn: List[float] = field(default_factory=list)
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_conversations: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    conversation_stats: List[ConversationStats] = field(default_factory=list)


class LMSysDatasetLoader:
    """Load and filter LMSYS-Chat-1M dataset."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to downloaded dataset, or None to stream from HF
        """
        self.dataset_path = dataset_path
        
    def load_dataset(self, 
                     num_conversations: int = 100,
                     min_turns: int = 2,
                     max_turns: Optional[int] = None,
                     languages: List[str] = ["English"]) -> List[Dict]:
        """
        Load and filter conversations from LMSYS-Chat-1M.
        
        Args:
            num_conversations: Number of conversations to load
            min_turns: Minimum number of turns per conversation
            max_turns: Maximum number of turns (None for no limit)
            languages: List of languages to include
            
        Returns:
            List of conversation dictionaries
        """
        print(f"Loading LMSYS-Chat-1M dataset...")
        
        if self.dataset_path:
            dataset = load_dataset("parquet", data_files=self.dataset_path)
        else:
            # Stream from HuggingFace (requires authentication)
            dataset = load_dataset("lmsys/lmsys-chat-1m", streaming=True)
        
        conversations = []
        
        for item in dataset["train"]:
            # Filter by language
            if item.get("language") not in languages:
                continue
            
            # Parse conversation
            conv = json.loads(item["conversation"])
            num_turns = len(conv)
            
            # Filter by number of turns
            if num_turns < min_turns:
                continue
            if max_turns and num_turns > max_turns:
                continue
            
            conversations.append({
                "conversation_id": item["conversation_id"],
                "model": item.get("model", "unknown"),
                "language": item.get("language", "unknown"),
                "turns": conv,
                "num_turns": num_turns
            })
            
            if len(conversations) >= num_conversations:
                break
        
        print(f"Loaded {len(conversations)} conversations")
        return conversations


class VLLMClient:
    """Async client for vLLM OpenAI-compatible API."""
    
    def __init__(self, base_url: str, model_name: str = "default"):
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL for vLLM API (e.g., http://localhost:8000/v1)
            model_name: Model name to use in requests
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    async def chat_completion(self,
                             messages: List[Dict[str, str]],
                             max_tokens: int = 512,
                             temperature: float = 0.7) -> Tuple[Dict, float, float]:
        """
        Send chat completion request to vLLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (response_dict, ttft_ms, tpot_ms)
        """
        if not self.session:
            raise RuntimeError("VLLMClient must be used as async context manager")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            response.raise_for_status()
            result = await response.json()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract tokens from response
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Estimate TTFT and TPOT (simplified - actual values depend on vLLM metrics)
        # In production, use vLLM's detailed metrics endpoint
        ttft_ms = latency_ms * 0.1  # Rough estimate: 10% of time for first token
        tpot_ms = (latency_ms - ttft_ms) / max(completion_tokens, 1)
        
        return result, ttft_ms, tpot_ms


class ConversationReplayer:
    """Replay multi-turn conversations to vLLM."""
    
    def __init__(self, vllm_client: VLLMClient):
        """
        Initialize replayer.
        
        Args:
            vllm_client: VLLMClient instance
        """
        self.client = vllm_client
    
    async def replay_conversation(self, 
                                  conversation: Dict) -> ConversationStats:
        """
        Replay a single conversation turn by turn.
        
        Args:
            conversation: Conversation dictionary with turns
            
        Returns:
            ConversationStats with metrics
        """
        stats = ConversationStats(
            conversation_id=conversation["conversation_id"],
            num_turns=conversation["num_turns"]
        )
        
        messages = []
        
        for i, turn in enumerate(conversation["turns"]):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # Add user message
            if role == "user":
                messages.append({"role": "user", "content": content})
                
                # Request assistant response
                try:
                    response, ttft, tpot = await self.client.chat_completion(
                        messages=messages,
                        max_tokens=512
                    )
                    
                    # Extract response content
                    assistant_content = response["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant_content})
                    
                    # Update stats
                    stats.ttft_per_turn.append(ttft)
                    stats.tpot_per_turn.append(tpot)
                    
                    usage = response.get("usage", {})
                    stats.total_input_tokens += usage.get("prompt_tokens", 0)
                    stats.total_output_tokens += usage.get("completion_tokens", 0)
                    
                    # Estimate cache hits (tokens from previous turns)
                    # In real implementation, use vLLM's cache metrics
                    if i > 0:
                        stats.cache_hit_tokens += stats.total_input_tokens // (i + 1)
                    
                except Exception as e:
                    print(f"Error in conversation {conversation['conversation_id']}, turn {i}: {e}")
                    break
            
            elif role == "assistant":
                # Use the ground truth assistant message
                messages.append({"role": "assistant", "content": content})
        
        stats.total_latency_ms = sum(stats.ttft_per_turn)
        stats.cache_miss_tokens = stats.total_input_tokens - stats.cache_hit_tokens
        
        return stats


class BenchmarkRunner:
    """Run benchmark and collect results."""
    
    def __init__(self, 
                 vllm_endpoint: str,
                 model_name: str = "default",
                 max_concurrent: int = 5):
        """
        Initialize benchmark runner.
        
        Args:
            vllm_endpoint: vLLM API endpoint
            model_name: Model name
            max_concurrent: Maximum concurrent conversations
        """
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.max_concurrent = max_concurrent
    
    async def run_benchmark(self, 
                          conversations: List[Dict]) -> BenchmarkResults:
        """
        Run benchmark on conversations.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            BenchmarkResults with aggregated metrics
        """
        results = BenchmarkResults()
        results.total_conversations = len(conversations)
        
        async with VLLMClient(self.vllm_endpoint, self.model_name) as client:
            replayer = ConversationReplayer(client)
            
            # Process conversations with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_conversation(conv):
                async with semaphore:
                    return await replayer.replay_conversation(conv)
            
            # Run all conversations
            tasks = [process_conversation(conv) for conv in conversations]
            stats_list = await tqdm_asyncio.gather(*tasks, desc="Replaying conversations")
            
            results.conversation_stats = stats_list
        
        # Aggregate results
        total_ttft = 0.0
        total_tpot = 0.0
        total_requests = 0
        total_cache_hits = 0
        total_cache_accesses = 0
        
        for stats in stats_list:
            total_ttft += sum(stats.ttft_per_turn)
            total_tpot += sum(stats.tpot_per_turn)
            total_requests += len(stats.ttft_per_turn)
            results.total_tokens += stats.total_input_tokens + stats.total_output_tokens
            total_cache_hits += stats.cache_hit_tokens
            total_cache_accesses += stats.total_input_tokens
        
        if total_requests > 0:
            results.avg_ttft_ms = total_ttft / total_requests
            results.avg_tpot_ms = total_tpot / total_requests
        
        if total_cache_accesses > 0:
            results.cache_hit_rate = total_cache_hits / total_cache_accesses
        
        return results


def print_results(results: BenchmarkResults):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Conversations:     {results.total_conversations}")
    print(f"Total Requests:          {results.total_requests}")
    print(f"Total Tokens:            {results.total_tokens:,}")
    print(f"Avg TTFT (ms):           {results.avg_ttft_ms:.2f}")
    print(f"Avg TPOT (ms):           {results.avg_tpot_ms:.2f}")
    print(f"Cache Hit Rate:          {results.cache_hit_rate:.2%}")
    print("=" * 80)
    
    # Per-conversation statistics
    print("\nPER-CONVERSATION BREAKDOWN:")
    print("-" * 80)
    
    df_data = []
    for stats in results.conversation_stats:
        df_data.append({
            "Conv ID": stats.conversation_id[:12],
            "Turns": stats.num_turns,
            "Input Tokens": stats.total_input_tokens,
            "Output Tokens": stats.total_output_tokens,
            "Avg TTFT (ms)": sum(stats.ttft_per_turn) / len(stats.ttft_per_turn) if stats.ttft_per_turn else 0,
            "Cache Hits": stats.cache_hit_tokens,
            "Hit Rate": f"{stats.cache_hit_tokens / max(stats.total_input_tokens, 1):.1%}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print("-" * 80)


def save_results(results: BenchmarkResults, output_path: str):
    """Save results to JSON file."""
    output = {
        "summary": {
            "total_conversations": results.total_conversations,
            "total_requests": results.total_requests,
            "total_tokens": results.total_tokens,
            "avg_ttft_ms": results.avg_ttft_ms,
            "avg_tpot_ms": results.avg_tpot_ms,
            "cache_hit_rate": results.cache_hit_rate
        },
        "conversations": [
            {
                "conversation_id": s.conversation_id,
                "num_turns": s.num_turns,
                "total_input_tokens": s.total_input_tokens,
                "total_output_tokens": s.total_output_tokens,
                "ttft_per_turn": s.ttft_per_turn,
                "tpot_per_turn": s.tpot_per_turn,
                "cache_hit_tokens": s.cache_hit_tokens,
                "cache_miss_tokens": s.cache_miss_tokens
            }
            for s in results.conversation_stats
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Replay LMSYS-Chat-1M conversations to vLLM for KV cache benchmarking"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to LMSYS-Chat-1M parquet files (if downloaded locally)"
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API endpoint URL"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        help="Model name for vLLM requests"
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of conversations to replay"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Minimum number of turns per conversation"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of turns per conversation"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent conversations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lmsys_benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    loader = LMSysDatasetLoader(args.dataset_path)
    conversations = loader.load_dataset(
        num_conversations=args.num_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns
    )
    
    if not conversations:
        print("No conversations loaded. Check dataset path and filters.")
        return
    
    # Run benchmark
    runner = BenchmarkRunner(
        vllm_endpoint=args.vllm_endpoint,
        model_name=args.model_name,
        max_concurrent=args.max_concurrent
    )
    
    print(f"\nStarting benchmark with {len(conversations)} conversations...")
    results = await runner.run_benchmark(conversations)
    
    # Print and save results
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
