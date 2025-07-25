# create_pilot_personas_v4_api.py

import json
import os
import random
import time
from openai import OpenAI

class Config:
    """設定を管理するクラス"""
    try:
        from google.colab import userdata
        API_KEY = userdata.get("OPENAI-KEY")
    except (ImportError, KeyError):
        API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    
    LLM_MODEL = "gpt-4o"
    INPUT_FILE = "QA_pairs_100_final.json"
    OUTPUT_DIR = "pilot_personas"
    NUM_PERSONAS = 10
    ANCHOR_CATEGORIES = ["ユーザーの名前", "誕生日", "出身地"]

def extract_core_info_with_api(client, category, answer_text):
    """
    【v4の核心機能 - あなたの提案】
    OpenAI APIを使い、answerの文章から核心的な情報（名前、日付など）を抽出する。
    """
    system_prompt = "あなたは、与えられた文章から特定の情報を正確に抽出する専門家です。"
    
    # カテゴリに応じた抽出指示
    if category == "ユーザーの名前":
        instruction = "文章に含まれるユーザーの名前だけを抽出してください。敬称（さん、くんなど）や読み仮名は含めないでください。"
    elif category == "誕生日":
        instruction = "文章に含まれる誕生日（日付）だけを「X月X日」の形式で抽出してください。"
    elif category == "出身地":
        instruction = "文章に含まれる出身地（都道府県や市町村名）だけを抽出してください。"
    else:
        return None # 不明なカテゴリ

    user_prompt = f"""
以下の文章から、指示に従って核心的な情報だけを抽出し、JSON形式で出力してください。

# 指示
{instruction}

# 文章
{answer_text}

# 出力形式
{{
  "core_info": "抽出した情報"
}}
"""
    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # 抽出タスクなので創造性は不要
        )
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data.get("core_info")
    except Exception as e:
        print(f"  - API抽出エラー: {e}")
        return None

def create_personas():
    """
    【v4】QAペアからAPIで核心情報を抽出し、クリーンなペルソナを生成する
    """
    print("--- パイロット・ペルソナ生成 (v4 - API抽出) を開始します ---")
    client = OpenAI(api_key=Config.API_KEY)

    try:
        with open(Config.INPUT_FILE, 'r', encoding='utf-8') as f:
            all_qa_pairs = json.load(f)
    except FileNotFoundError:
        print(f"❌ エラー: 入力ファイル '{Config.INPUT_FILE}' が見つかりません。")
        return

    anchor_facts_pool = {category: [] for category in Config.ANCHOR_CATEGORIES}
    other_facts_pool = []
    for qa in all_qa_pairs:
        category = qa.get('category')
        if category in Config.ANCHOR_CATEGORIES:
            anchor_facts_pool[category].append(qa)
        else:
            other_facts_pool.append(qa)

    personas = []
    for i in range(Config.NUM_PERSONAS):
        persona_data = {
            "persona_id": i + 1,
            "persona": {
                "profile": {},
                "source_anchor_facts": [],
            },
            "other_facts": []
        }

        for category in Config.ANCHOR_CATEGORIES:
            if anchor_facts_pool[category]:
                fact_qa = random.choice(anchor_facts_pool[category])
                
                print(f"ペルソナ{i+1}: 「{category}」の情報を抽出中...")
                # APIを使って核心情報を抽出
                core_info = extract_core_info_with_api(client, fact_qa['category'], fact_qa['answer'])
                
                if core_info:
                    key_map = {"ユーザーの名前": "name", "誕生日": "birthday", "出身地": "from"}
                    persona_data["persona"]["profile"][key_map[category]] = core_info
                else:
                    print(f"  - 抽出に失敗しました。")
                    
                persona_data["persona"]["source_anchor_facts"].append(fact_qa)
                anchor_facts_pool[category].remove(fact_qa)
                time.sleep(1) # APIへの負荷軽減
        
        personas.append(persona_data)

    random.shuffle(other_facts_pool)
    for i, fact in enumerate(other_facts_pool):
        persona_index = i % Config.NUM_PERSONAS
        personas[persona_index]["other_facts"].append(fact)

    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    for persona in personas:
        output_filename = os.path.join(Config.OUTPUT_DIR, f"persona_{persona['persona_id']:02d}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)
            
    print(f"\n✅ 全てのペルソナプロファイルを '{Config.OUTPUT_DIR}' ディレクトリに出力しました。")
    print("\n🎉 パイロット・ペルソナ生成 (v4 - API抽出) が完了しました！")

if __name__ == "__main__":
    create_personas()
