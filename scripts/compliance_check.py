# -*- coding: utf-8 -*-
"""DKE Guide-for-Authors compliance checks against the manuscript."""
import re
import sys

TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()

m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
body = m.group(1)
# strip latex commands/math for a fair word count
clean = re.sub(r'\$[^$]*\$', 'X', body)
clean = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', clean)
clean = re.sub(r'[{}~]', ' ', clean)
words = clean.split()
print('abstract words:', len(words))
print('citations in abstract:', len(re.findall(r'\\cite', body)))
print('--- abstract text (cleaned) ---')
print(' '.join(words)[:1200])

kw = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', t, re.S)
print('\nkeywords raw:', kw.group(1).strip() if kw else 'NOT FOUND')
kwcount = len(re.split(r'\\\\|;', kw.group(1))) if kw else 0
print('keyword count (approx):', kwcount)

print('\nGenAI declaration section present:',
      'Declaration of generative AI' in t or 'generative AI' in t)
print('CRediT present:', 'CRediT' in t)
print('Declaration of interests present:', 'Declaration of' in t or 'competing' in t.lower())
print('Data availability present:', 'Data availability' in t or 'data availab' in t.lower())
print('Funding present:', 'unding' in t)
print('Anonymous author block present:', 'Anonymous' in t)
