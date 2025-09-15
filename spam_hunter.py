"""
Viorazu.xai-spam-hunter
X (Twitter) Spam Detection & Originality Checker

© 2025 Viorazu. All rights reserved.

Author: Viorazu.
Created: 2025-09-15
Purpose: リライト記事とBotによるプラットフォーム汚染を防ぐ

Commercial use requires prior permission.
Contact: X @viorazu9134
"""

import re
import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError, jwt
import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
JWT_SECRET = os.getenv("JWT_SECRET", "x7b9k3m8p2q5w4r6t1y0u8i2o3j5h9g4")
JWT_ALGORITHM = "HS256"
THRESHOLD = 0.9

class CheckRequest(BaseModel):
    content: str

async def get_api_key(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="無効なトークン")
        return token
    except JWTError:
        raise HTTPException(status_code=401, detail="無効なトークン")

class SpamHunter:
    def __init__(self):
        self.mecab = MeCab.Tagger()
        self.tfidf = TfidfVectorizer(max_features=500, tokenizer=self._tokenize, token_pattern=None)
        self.history = {}

    def _tokenize(self, text: str) -> List[str]:
        words = []
        parsed = self.mecab.parse(text)
        for line in parsed.split('\n'):
            if line in ('EOS', ''):
                break
            try:
                word, features = line.split('\t')
                if features.split(',')[0] in ['名詞', '動詞', '形容詞']:
                    words.append(word)
            except:
                continue
        return words

    def calculate_score(self, content: str) -> float:
        hashtags = [tag.lower() for tag in re.findall(r'#\w+', content)]
        common_tags = ['#ai', '#副業', '#投資']
        hashtag_score = 0.1 * sum(1 for tag in hashtags if tag not in common_tags)
        base_score = len(self._tokenize(content)) / 20
        return min(base_score + hashtag_score, 1.0)

    def is_likely_bot(self, content: str, user_id: str = "default") -> bool:
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append(content)
        self.history[user_id] = self.history[user_id][-10:]

        patterns = [re.sub(r'\s+', ' ', t.strip()) for t in self.history[user_id]]
        unique_ratio = len(set(patterns)) / len(patterns) if patterns else 1.0
        high_freq = len(self.history[user_id]) > 8

        high_sim = False
        if len(self.history[user_id]) > 1:
            try:
                tfidf = self.tfidf.fit_transform(self.history[user_id]).toarray()
                sims = np.mean([np.dot(tfidf[i], tfidf[j]) for i in range(len(tfidf)) for j in range(i + 1, len(tfidf))])
                high_sim = sims > 0.8
            except:
                pass

        url_count = len(re.findall(r'http[s]?://', content))
        emoji_count = len(re.findall(r'[😄🚀💰💸🔥]', content))
        spam_pattern = url_count > 5 or emoji_count > 10

        return unique_ratio < 0.5 or high_freq or high_sim or spam_pattern

    def get_badge(self, score: float) -> str:
        if score >= 0.90:
            return "ORIGINAL_GOLD"
        if score >= 0.85:
            return "ORIGINAL_SILVER"
        if score >= 0.75:
            return "ORIGINAL_BRONZE"
        return "NONE"

hunter = SpamHunter()

@app.post("/check")
async def check_spam(request: CheckRequest, token: str = Depends(get_api_key)):
    try:
        score = hunter.calculate_score(request.content)
        is_bot = hunter.is_likely_bot(request.content)
        badge = hunter.get_badge(score)
        return {
            "score": round(score, 2),
            "is_bot": is_bot,
            "badge": badge
        }
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise HTTPException(status_code=500, detail="処理エラー")

@app.post("/token")
async def login():
    from datetime import datetime, timedelta
    payload = {"sub": "user_001", "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
