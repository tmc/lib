import argparse,json,random
import tokenizers
if tokenizers.__version__ != "0.22.2":
 raise RuntimeError("fixture generation requires tokenizers==0.22.2")
from pathlib import Path
from tokenizers import Tokenizer,models,pre_tokenizers,normalizers,processors,decoders,AddedToken,Regex
root=Path(__file__).resolve().parent
args=argparse.ArgumentParser()
args.add_argument('--qwen-dir',type=Path)
qwen_dir=args.parse_args().qwen_dir
rng=random.Random(91023)
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:]
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+len(cs)-len(list(range(33,127))+list(range(161,173))+list(range(174,256))))
alphabet=dict(zip(bs,map(chr,cs)))
vocab={alphabet[b]:(b*73)%256 for b in range(256)};merges=[]
def merge(a,b):
 assert a in vocab and b in vocab
 word=a+b
 if word not in vocab:vocab[word]=len(vocab)
 merges.append((a,b))
for a,b in [('a','a'),('aa','a'),('a','b'),('b','c'),('ab','c'),(alphabet[32],'a'),(alphabet[32]+'a','b'),(alphabet[195],alphabet[169])]:merge(a,b)
texts=['','hello world','  hello   world  ','\r\n\n  \t a\n\n','a\u00a0b\u2003c','cafe\u0301 café','你好世界 😀🙂','abcABCaaabcbcaaaaaaa','1234567 12.34 Ⅻ²','WE\'RE I\'m it\'S he\'ſ','<BOS>hello<EOS>','ABC AB A<tag>','e\u0301x xe\u0301 \u0301','\0\u0085\u2028\u2029\ufeff','𐐀𐐨 \U0001fae8','𝟙𝟚𝟛 ⅣⅩ ๙٩']
chars=list(' abcxyzABC0123456789\n\r\t\'!.,<>')+list('é\u0301界🙂ſ\u00a0\u2003\u0085\u2028Ⅻ²')
for _ in range(160):texts.append(''.join(rng.choice(chars) for _ in range(rng.randrange(1,90))))
qwen=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
llama=qwen.replace(r'|\p{N}|',r'|\p{N}{1,3}|')
records=[]
for name in ['gpt2','gpt2-prefix','qwen','llama','llama-multi','raw']:
 t=Tokenizer(models.BPE(vocab=vocab,merges=merges))
 t.normalizer=normalizers.NFC()
 if name in ['qwen','llama','llama-multi']:
  t.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(qwen if name=='qwen' else llama),behavior='isolated'),pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
 else:t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=name=='gpt2-prefix',use_regex=name!='raw')
 t.decoder=decoders.ByteLevel()
 t.add_special_tokens([AddedToken(s,normalized=False,special=True) for s in ['<BOS>','<EOS>','<tag>','AB','ABC','e\u0301']])
 if name.startswith('llama'):
  t.post_processor=processors.TemplateProcessing(single='<BOS> <tag> $A <EOS> <tag>' if name=='llama-multi' else '<BOS> $A <EOS>',special_tokens=[('<BOS>',t.token_to_id('<BOS>')),('<EOS>',t.token_to_id('<EOS>')),('<tag>',t.token_to_id('<tag>'))])
 t.save(str(root/(name+'.json')))
 cases=[]
 for s in texts:
  ids=t.encode(s).ids;cases.append({'text':s,'ids':ids,'decoded':t.decode(ids,skip_special_tokens=True)})
 decoding=[]
 for data in [b'\xff\xff',b'\xef\xbb',b'\xf0\x9f\x98\x80',b'\xed\xa0\x80',b'\xe0\x80\x80']+[bytes(rng.randrange(256) for _ in range(rng.randrange(1,20))) for _ in range(60)]:
  ids=[vocab[alphabet[b]] for b in data]
  decoding.append({'ids':ids,'decoded':t.decode(ids,skip_special_tokens=True),'bytes':list(data)})
 records.append({'file':name+'.json','cases':cases,'decoding':decoding})
(root/'reference.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':'))+'\n')
if qwen_dir is not None:
 real=Tokenizer.from_file(str(qwen_dir/'tokenizer.json'))
 real_texts=texts+['<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n','<|endoftext|>  test']
 real_cases=[{'text':s,'ids':real.encode(s).ids,'decoded':real.decode(real.encode(s).ids,skip_special_tokens=True)} for s in real_texts]
 (qwen_dir/'reference.json').write_text(json.dumps(real_cases,ensure_ascii=False,indent=2)+'\n')
print('portable pipelines',len(records),'encode cases',sum(len(r['cases']) for r in records),'decode cases',sum(len(r['decoding']) for r in records))
