# word_similarity_app.py
"""
Streamlit 日本語版：分布仮説アクティビティ用 Web アプリ
======================================================

* **参加者ページ**（デフォルト `/`）
    * ニックネームと好きな単語を入力すると、講師が設定した"秘密の単語"との類似度スコアが表示されます。
* **講師ページ**（`/teacher`）
    * 秘密の単語の設定／変更
    * 参加者の回答をリアルタイムで類似度順ランキング表示

Streamlit の複数ページ機能（`st.Page` / `st.navigation`）を使っていますが、サイドバーのナビゲーションは CSS で非表示にしているため、受講者側には講師ページへのリンクは見えません。講師は URL `/teacher` を手入力してください。

実行例：
```bash
export OPENAI_API_KEY="sk-..."
streamlit run word_similarity_app.py  # http://localhost:8501/ 参加者ページ
# 講師は http://localhost:8501/teacher にアクセス
```
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import openai
import pandas as pd
import streamlit as st

# ------------------------------ 設定 ------------------------------ #
DATA_DIR = Path(".data")
DATA_DIR.mkdir(exist_ok=True)
RESPONSES_CSV = DATA_DIR / "responses.csv"
SECRET_FILE = DATA_DIR / "secret_word.json"
EMBED_MODEL = "text-embedding-3-small"  # 1536 次元

OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または Streamlit の Secret で指定してください。")
    st.stop()

# ---- OpenAI SDK version check ----
# v1.x では `openai.OpenAI`, v0.x では `openai.api_key` スタイル。
try:
    client = openai.OpenAI(api_key=OPENAI_KEY)  # type: ignore[attr-defined]
    _use_client = True  # 新 SDK
except AttributeError:
    openai.api_key = OPENAI_KEY  # 旧 SDK
    _use_client = False

st.set_page_config(page_title="単語類似度チャレンジ", page_icon="🧠", initial_sidebar_state="collapsed")

# ---- サイドバーを完全に非表示にする ----
st.markdown(
    """<style>
        section[data-testid='stSidebar'] {display: none !important;}
        div[data-testid='stSidebarNav'] {display: none !important;}
    </style>""",
    unsafe_allow_html=True,
)

# ------------------------------ 共通関数 ------------------------------ #

def get_embedding(text: str, model: str = EMBED_MODEL) -> np.ndarray:  # type: ignore[override]
    """OpenAI Embedding を取得し numpy 配列で返す。SDK v1 / v0 両対応"""
    if _use_client:  # 新 SDK (>=1.0)
        resp = client.embeddings.create(input=text, model=model)
        embedding = resp.data[0].embedding  # type: ignore[index]
    else:  # 旧 SDK (<1.0)
        resp = openai.Embedding.create(input=[text], model=model)  # type: ignore[attr-defined]
        embedding = resp["data"][0]["embedding"]  # type: ignore[index]
    return np.asarray(embedding, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_secret_word() -> str | None:
    if SECRET_FILE.exists():
        with SECRET_FILE.open() as f:
            data = json.load(f)
        return data.get("secret")
    return None


def save_secret_word(word: str) -> None:
    with SECRET_FILE.open("w") as f:
        json.dump({"secret": word}, f, ensure_ascii=False)


@st.cache_data(show_spinner=False)
def load_responses() -> pd.DataFrame:  # pragma: no cover
    if RESPONSES_CSV.exists():
        return pd.read_csv(RESPONSES_CSV)
    return pd.DataFrame(columns=["nickname", "word", "similarity"])


def add_response(nickname: str, word: str, sim: float) -> None:
    df = load_responses()
    df.loc[len(df)] = [nickname, word, sim]
    df.to_csv(RESPONSES_CSV, index=False)
    load_responses.clear()

# ------------------------------ ページ定義 ------------------------------ #

def participant_view() -> None:
    """参加者用 UI"""
    st.header("🔍 類似度チャレンジ – 単語を当てよう！")

    secret_word = load_secret_word()
    if secret_word is None:
        st.warning("まだ講師がセッションを開始していません。少々お待ちください。")
        return

    nickname = st.text_input("ニックネーム（ランキングに表示されます）")
    guess = st.text_input("あなたが近いと思う単語を入力：")

    if st.button("送信", use_container_width=True):
        if not nickname or not guess:
            st.error("ニックネームと単語の両方を入力してください。")
            return

        with st.spinner("類似度を計算中..."):
            try:
                emb_secret = get_embedding(secret_word)
                emb_guess = get_embedding(guess)
                similarity = cosine_sim(emb_secret, emb_guess)
            except openai.OpenAIError as e:
                st.error(f"OpenAI API エラー: {e}")
                return

        add_response(nickname.strip(), guess.strip(), round(similarity, 4))
        st.success(f"類似度スコア: {similarity:.4f} (1 に近いほど類似) ✨")
        st.balloons()

    # ランキング表示
    st.subheader("🏆 現在のトップ 10")
    df = load_responses()
    if not df.empty:
        top = df.sort_values("similarity", ascending=False).head(10).reset_index(drop=True)
        st.table(top)


def teacher_view() -> None:
    """講師用 UI"""
    st.header("🧑‍🏫 講師コントロールパネル")

    secret_word = load_secret_word()
    if secret_word is None:
        st.info("秘密の単語が未設定です。下の入力欄に設定してください。")
        new_secret = st.text_input("秘密の単語", key="secret_input")
        if st.button("保存") and new_secret:
            save_secret_word(new_secret.strip())
            st.success("秘密の単語を保存しました！ 参加者が回答できます。")
            secret_word = new_secret.strip()
            st.rerun()
    else:
        st.success("✅ 秘密の単語は設定済み（非表示）")
        with st.expander("秘密の単語を変更する"):
            new_secret = st.text_input("新しい秘密の単語", key="new_secret_input")
            if st.button("更新") and new_secret:
                save_secret_word(new_secret.strip())
                # 回答リセット
                if RESPONSES_CSV.exists():
                    RESPONSES_CSV.unlink()
                load_responses.clear()
                st.success("秘密の単語を更新し、過去の回答をクリアしました。")
                st.rerun()

    st.subheader("📊 ランキング（上位 20 件）")
    df = load_responses()
    if df.empty:
        st.write("まだ回答がありません。")
    else:
        top = df.sort_values("similarity", ascending=False).head(20).reset_index(drop=True)
        st.table(top)

# ------------------------------ ナビゲーション ------------------------------ #

participant_page = st.Page(participant_view, title="参加者", icon="🎯", default=True)
teacher_page = st.Page(teacher_view, title="teacher", icon="🧑‍🏫")

pg = st.navigation([participant_page, teacher_page])
pg.run()

# ------------------------------ フッター ------------------------------ #
with st.container():
    st.markdown("---")
    st.caption(
        "作成: ❤️ [Streamlit](https://streamlit.io) + OpenAI Embeddings  ／ ソースコードは教育目的で公開しています。\n"
        "URL `/teacher` を直接入力すると講師ページにアクセスできます（サイドバーは非表示のまま）。"
    )
