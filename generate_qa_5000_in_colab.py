# generate_qa_5000_in_colab.py
import os
import json
import time
import math
from openai import OpenAI
from tqdm.notebook import tqdm  # Colabのtqdmに変更

# ---------------------------------
# 1. 設定と計画
# ---------------------------------
class Config:
    """設定を管理するクラス"""
    try:
        from google.colab import userdata
        API_KEY = userdata.get("OPENAI-KEY")
    except (ImportError, KeyError):
        API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

    LLM_MODEL = "gpt-4o"
    TEMPERATURE = 0.95
    # バッチファイルの出力先ディレクトリ
    BATCH_OUTPUT_DIR = "batches"

def get_full_generation_plan(total_items=5000):
    """5,000件規模の全体計画を生成する"""
    # 各Tierの比率（例）
    tier_ratios = {
        "1": 0.25,  # 1250件
        "2": 0.25,  # 1250件
        "3": 0.30,  # 1500件
        "4": 0.20,  # 1000件
    }
    
    # 各カテゴリの定義
    categories = {
        "1": ["ユーザーの名前", "人生の目標", "価値観", "アイデンティティ"],
        "2": ["恐怖症", "アレルギー", "トラウマ"],
        "3": ["誕生日", "出身地", "ペットの名前", "家族・友人の名前"],
        "4": ["好きな作家・音楽・映画", "趣味・休日の過ごし方", "好きな食べ物・飲み物", "嫌いな食べ物・飲み物"]
    }
    
    full_plan = []
    for tier, ratio in tier_ratios.items():
        num_tier_items = int(total_items * ratio)
        for i in range(num_tier_items):
            # カテゴリをランダムに選択
            category = categories[tier][i % len(categories[tier])]
            full_plan.append((tier, category))
            
    # 全体の件数がtotal_itemsになるように調整
    return full_plan[:total_items]

# ---------------------------------
# 2. QAペア生成のコアロジック (v4から流用)
# ---------------------------------
def generate_qa_pair(client, category):
    """高品質なQAペアを1つ生成する（プロンプトは検証済みのものをそのまま使用）"""
    system_prompt = "あなたは、人間らしい記憶に関する、高品質で創造的なデータセットを生成する専門家です。"
    user_prompt = f"""
以下の厳格なルールに従い、「{category}」に関する高品質なQAペアを1つ生成してください。
# ルール
1.  **回答(Answer)の要件**:
    - アシスタントがユーザーの記憶を確認する形式で、必ず「はい、...」で始めてください。
    - カテゴリに合致する中核的な事実（コア・ファクト）を明確に含んでください。
    - コア・ファクトを裏付ける、具体的で推測困難なエピソードを必ず含めてください。
    - 全体で1〜3文程度の簡潔な文章にしてください。
2.  **質問(Question)の要件**:
    - ユーザーがアシスタントの記憶力を試すような、自然な問いかけにしてください。
    - 【最重要】質問は、あなたが上記ルール1で生成する回答文の中の「コア・ファクト」を、直接的に問う内容でなければなりません。
    - **質問文の主題と疑問詞（例：「何」「誰」「どんな」）は、必ず「コア・ファクト」そのものに向けられていなければなりません。関連する他の情報（例：イベント名、場所、時間）を問う質問は許可しません。**
    - 質問と回答は、論理的に完全に一貫していなければなりません。
# 出力形式
- 必ず以下のJSON形式で出力してください。
{{
  "category": "{category}",
  "question": "生成した質問",
  "answer": "生成した回答"
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
            temperature=Config.TEMPERATURE,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  - APIエラー発生: {e}")
        return None

# ---------------------------------
# 3. Colab用バッチ実行・自動保存エンジン
# ---------------------------------
def run_batch_generation(total_items=5000, batch_size=100):
    """Colab環境で、中断・再開可能なバッチ生成を実行する"""
    
    print("--- QAペアの大量生成を開始します ---")
    
    client = OpenAI(api_key=Config.API_KEY)
    
    # 出力ディレクトリの作成
    if not os.path.exists(Config.BATCH_OUTPUT_DIR):
        os.makedirs(Config.BATCH_OUTPUT_DIR)
        print(f"出力ディレクトリ '{Config.BATCH_OUTPUT_DIR}' を作成しました。")

    # 全体計画の取得
    full_plan = get_full_generation_plan(total_items)
    num_batches = math.ceil(total_items / batch_size)

    print(f"全体計画: {total_items}件 / バッチサイズ: {batch_size}件 / 合計バッチ数: {num_batches}件")
    print("-" * 30)

    for i in range(num_batches):
        batch_num = i + 1
        output_filename = os.path.join(Config.BATCH_OUTPUT_DIR, f"batch_{batch_num:03d}.json")

        # 【重要】チェックポイント機能：既にファイルが存在すればスキップ
        if os.path.exists(output_filename):
            print(f"☑ バッチ {batch_num}/{num_batches} は既に存在するため、スキップします。")
            continue

        print(f"▶️  バッチ {batch_num}/{num_batches} の生成を開始...")
        
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_plan = full_plan[start_index:end_index]
        
        batch_dataset = []
        for tier, category in tqdm(batch_plan, desc=f"バッチ {batch_num} 生成中"):
            qa_pair = None
            for _ in range(3): # 3回までリトライ
                qa_pair = generate_qa_pair(client, category)
                if qa_pair and qa_pair.get("question") and qa_pair.get("answer"):
                    break
                time.sleep(5)

            if qa_pair:
                batch_dataset.append({
                    'tier': int(tier),
                    'category': category,
                    'question': qa_pair.get('question'),
                    'answer': qa_pair.get('answer')
                })
        
        # バッチが完了するたびに、結果をファイルに書き出す
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(batch_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ バッチ {batch_num}/{num_batches} の生成が完了し、'{output_filename}' に保存しました。({len(batch_dataset)}件)\n")
        time.sleep(10) # APIへの負荷軽減のため小休止

    print("🎉 全てのバッチ生成が完了しました！")
    print(f"次に、merge_batches.py を実行して、'{Config.BATCH_OUTPUT_DIR}' 内のファイルを結合してください。")


# ---- 実行 ----
if __name__ == "__main__":
    # ★★★ 本番用の設定に戻しました ★★★
    run_batch_generation(total_items=5000, batch_size=100)
