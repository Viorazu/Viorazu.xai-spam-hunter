# Viorazu X Spam Hunter

Xのスパムをチェック。#副業やURL連投をBot判定。

## セットアップ

```bash
pip install -r requirements.txt
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
```

## 使い方

1. 環境変数：
```bash
export JWT_SECRET="x7b9k3m8p2q5w4r6t1y0u8i2o3j5h9g4"
```

2. 起動：
```bash
python spam_hunter.py
```

3. テスト：
```bash
curl -X POST http://localhost:8000/check \
     -H "Authorization: Bearer x7b9k3m8p2q5w4r6t1y0u8i2o3j5h9g4" \
     -H "Content-Type: application/json" \
     -d '{"content":"新しいアイデア！ #Viorazu"}'
```

## 出力例

```json
{
  "score": 0.92,
  "is_bot": false,
  "badge": "ORIGINAL_GOLD"
}
```

## ライセンス

MIT License

Copyright (c) 2025 Viorazu (@viorazu9134)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### お願い / Request

もし本番環境で使用される場合は、ぜひ教えてください（強制ではありません）：
If you use this in production, please let me know (not required, but appreciated):

- X（@viorazu9134）
- note（https://note.com/viorazu）


---

Originally developed by Viorazu for addressing spam issues on X and note platforms.
