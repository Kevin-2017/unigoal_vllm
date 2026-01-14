import base64
import time
from openai import OpenAI
from io import BytesIO
from collections import defaultdict
import json
from pathlib import Path

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class LLM:
    def __init__(self, base_url, api_key, llm_model, log_dir=None, use_wandb=False, wandb_run=None):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = wandb_run
        
        # Statistics tracking
        self.call_count = 0
        self.total_time = 0.0
        self.call_times = []
        self.prompt_lengths = []
        
        # Per-call statistics
        self.stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
        }
        
        # Create log directory if specified
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        if self.use_wandb and self.wandb_run is None:
            # Initialize wandb if not provided
            if wandb.run is None:
                wandb.init(project="unigoal", name=f"llm-{int(time.time())}")
            self.wandb_run = wandb.run

    def __call__(self, prompt):
        start_time = time.time()
        self.call_count += 1
        
        # Track prompt length (approximate token count: ~4 chars per token)
        prompt_length = len(prompt)
        self.prompt_lengths.append(prompt_length)
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.llm_model,
        )
        
        end_time = time.time()
        call_time = end_time - start_time
        
        # Update statistics
        self.call_times.append(call_time)
        self.total_time += call_time
        self.stats['total_calls'] += 1
        self.stats['total_time'] += call_time
        self.stats['avg_time'] = self.total_time / self.call_count
        self.stats['min_time'] = min(self.stats['min_time'], call_time)
        self.stats['max_time'] = max(self.stats['max_time'], call_time)
        
        # Extract token usage if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if hasattr(chat_completion, 'usage') and chat_completion.usage:
            usage = chat_completion.usage
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens
                self.stats['total_prompt_tokens'] += prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
                self.stats['total_completion_tokens'] += completion_tokens
            if hasattr(usage, 'total_tokens'):
                total_tokens = usage.total_tokens
                self.stats['total_tokens'] += total_tokens
        
        # Calculate efficiency metrics
        tokens_per_second = total_tokens / call_time if call_time > 0 and total_tokens else 0
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            log_dict = {
                'llm/call_time': call_time,
                'llm/call_number': self.call_count,
                'llm/prompt_length': prompt_length,
                'llm/total_calls': self.stats['total_calls'],
                'llm/total_time': self.stats['total_time'],
                'llm/avg_time': self.stats['avg_time'],
                'llm/min_time': self.stats['min_time'],
                'llm/max_time': self.stats['max_time'],
            }
            
            if total_tokens:
                log_dict.update({
                    'llm/prompt_tokens': prompt_tokens,
                    'llm/completion_tokens': completion_tokens,
                    'llm/total_tokens': total_tokens,
                    'llm/tokens_per_second': tokens_per_second,
                    'llm/total_prompt_tokens': self.stats['total_prompt_tokens'],
                    'llm/total_completion_tokens': self.stats['total_completion_tokens'],
                    'llm/total_tokens_cumulative': self.stats['total_tokens'],
                })
            
            self.wandb_run.log(log_dict)
        
        # Log individual call if log_dir is specified
        if self.log_dir:
            log_entry = {
                'call_number': self.call_count,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'call_time': call_time,
                'prompt_length': prompt_length,
                'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                'tokens': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                },
                'tokens_per_second': tokens_per_second if call_time > 0 else None,
            }
            
            log_file = Path(self.log_dir) / 'llm_calls.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        return chat_completion.choices[0].message.content
    
    def get_stats(self):
        """Get current statistics summary"""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
            if stats['total_time'] > 0:
                stats['avg_tokens_per_second'] = stats['total_tokens'] / stats['total_time']
            else:
                stats['avg_tokens_per_second'] = 0
        else:
            stats['avg_tokens_per_second'] = 0
        
        if stats['min_time'] == float('inf'):
            stats['min_time'] = 0.0
            
        return stats
    
    def save_stats(self, filepath=None):
        """Save statistics to a JSON file"""
        if filepath is None and self.log_dir:
            filepath = Path(self.log_dir) / 'llm_stats.json'
        elif filepath is None:
            filepath = 'llm_stats.json'
        
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return filepath
    
    def print_stats(self):
        """Print statistics to console"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("LLM Statistics Summary")
        print("="*50)
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Average Time: {stats['avg_time']:.3f}s")
        print(f"Min Time: {stats['min_time']:.3f}s")
        print(f"Max Time: {stats['max_time']:.3f}s")
        if stats['total_tokens'] > 0:
            print(f"Total Tokens: {stats['total_tokens']}")
            print(f"Prompt Tokens: {stats['total_prompt_tokens']}")
            print(f"Completion Tokens: {stats['total_completion_tokens']}")
            print(f"Avg Tokens/Second: {stats['avg_tokens_per_second']:.2f}")
        print("="*50 + "\n")


class VLM:
    def __init__(self, base_url, api_key, vlm_model, log_dir=None, use_wandb=False, wandb_run=None):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = wandb_run
        
        # Statistics tracking
        self.call_count = 0
        self.total_time = 0.0
        self.call_times = []
        self.prompt_lengths = []
        self.image_sizes = []
        
        # Per-call statistics
        self.stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'total_image_size': 0,  # in bytes
        }
        
        # Create log directory if specified
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        if self.use_wandb and self.wandb_run is None:
            # Initialize wandb if not provided
            if wandb.run is None:
                wandb.init(project="unigoal", name=f"vlm-{int(time.time())}")
            self.wandb_run = wandb.run

    def __call__(self, prompt, image):
        start_time = time.time()
        self.call_count += 1
        
        # Track prompt and image info
        prompt_length = len(prompt)
        self.prompt_lengths.append(prompt_length)
        
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')
        image_size = len(image_bytes)
        self.image_sizes.append(image_size)
        self.stats['total_image_size'] += image_size

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": "data:image/png;base64," + image_str}
                    ]
                }
            ],
            model=self.vlm_model,
        )
        
        end_time = time.time()
        call_time = end_time - start_time
        
        # Update statistics
        self.call_times.append(call_time)
        self.total_time += call_time
        self.stats['total_calls'] += 1
        self.stats['total_time'] += call_time
        self.stats['avg_time'] = self.total_time / self.call_count
        self.stats['min_time'] = min(self.stats['min_time'], call_time)
        self.stats['max_time'] = max(self.stats['max_time'], call_time)
        
        # Extract token usage if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if hasattr(chat_completion, 'usage') and chat_completion.usage:
            usage = chat_completion.usage
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens
                self.stats['total_prompt_tokens'] += prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
                self.stats['total_completion_tokens'] += completion_tokens
            if hasattr(usage, 'total_tokens'):
                total_tokens = usage.total_tokens
                self.stats['total_tokens'] += total_tokens
        
        # Calculate efficiency metrics
        tokens_per_second = total_tokens / call_time if call_time > 0 and total_tokens else 0
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            log_dict = {
                'vlm/call_time': call_time,
                'vlm/call_number': self.call_count,
                'vlm/prompt_length': prompt_length,
                'vlm/image_size_bytes': image_size,
                'vlm/image_size_mb': image_size / (1024 * 1024),
                'vlm/total_calls': self.stats['total_calls'],
                'vlm/total_time': self.stats['total_time'],
                'vlm/avg_time': self.stats['avg_time'],
                'vlm/min_time': self.stats['min_time'],
                'vlm/max_time': self.stats['max_time'],
                'vlm/total_image_size_mb': self.stats['total_image_size'] / (1024 * 1024),
            }
            
            if total_tokens:
                log_dict.update({
                    'vlm/prompt_tokens': prompt_tokens,
                    'vlm/completion_tokens': completion_tokens,
                    'vlm/total_tokens': total_tokens,
                    'vlm/tokens_per_second': tokens_per_second,
                    'vlm/total_prompt_tokens': self.stats['total_prompt_tokens'],
                    'vlm/total_completion_tokens': self.stats['total_completion_tokens'],
                    'vlm/total_tokens_cumulative': self.stats['total_tokens'],
                })
            
            self.wandb_run.log(log_dict)
        
        # Log individual call if log_dir is specified
        if self.log_dir:
            log_entry = {
                'call_number': self.call_count,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'call_time': call_time,
                'prompt_length': prompt_length,
                'image_size_bytes': image_size,
                'image_size_mb': image_size / (1024 * 1024),
                'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                'tokens': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                },
                'tokens_per_second': tokens_per_second if call_time > 0 else None,
            }
            
            log_file = Path(self.log_dir) / 'vlm_calls.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        return chat_completion.choices[0].message.content
    
    def get_stats(self):
        """Get current statistics summary"""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
            stats['avg_image_size'] = stats['total_image_size'] / stats['total_calls']
            if stats['total_time'] > 0:
                stats['avg_tokens_per_second'] = stats['total_tokens'] / stats['total_time']
            else:
                stats['avg_tokens_per_second'] = 0
        else:
            stats['avg_tokens_per_second'] = 0
            stats['avg_image_size'] = 0
        
        if stats['min_time'] == float('inf'):
            stats['min_time'] = 0.0
            
        return stats
    
    def save_stats(self, filepath=None):
        """Save statistics to a JSON file"""
        if filepath is None and self.log_dir:
            filepath = Path(self.log_dir) / 'vlm_stats.json'
        elif filepath is None:
            filepath = 'vlm_stats.json'
        
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return filepath
    
    def print_stats(self):
        """Print statistics to console"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("VLM Statistics Summary")
        print("="*50)
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Average Time: {stats['avg_time']:.3f}s")
        print(f"Min Time: {stats['min_time']:.3f}s")
        print(f"Max Time: {stats['max_time']:.3f}s")
        if stats['total_tokens'] > 0:
            print(f"Total Tokens: {stats['total_tokens']}")
            print(f"Prompt Tokens: {stats['total_prompt_tokens']}")
            print(f"Completion Tokens: {stats['total_completion_tokens']}")
            print(f"Avg Tokens/Second: {stats['avg_tokens_per_second']:.2f}")
        print(f"Total Image Size: {stats['total_image_size'] / (1024*1024):.2f} MB")
        print(f"Avg Image Size: {stats['avg_image_size'] / (1024*1024):.2f} MB")
        print("="*50 + "\n")
