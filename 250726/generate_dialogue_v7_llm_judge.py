# generate_dialogues_large_scale_v8_ultimate.py

import json
import os
import random
import time
import re
from openai import OpenAI
from tqdm.notebook import tqdm

class Config:
    try:
        from google.colab import userdata
        API_KEY = userdata.get("OPENAI-KEY")
    except (ImportError, KeyError):
        API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    LLM_MODEL = "gpt-4o"
    PERSONA_DIR = "pilot_personas" # 本番生成時は "personas" に変更
    OUTPUT_DIR = "pilot_dialogues" # 本番生成時は "dialogues" に変更
    NUM_TURNS = 50
    NUM_INJECTIONS = 2
    PRESENCE_PENALTY = 0.2

def create_dynamic_injection_prompt(persona_data, fact_to_inject, dialogue_history):
    profile = persona_data["persona"]["profile"]
    persona_summary_text = "\n".join([f"- {key}: {value}" for key, value in profile.items()])
    history_text = "\n".join([f"{turn['speaker']}: {turn['content']}" for turn in dialogue_history[-10:]])
    core_fact_to_inject = f"- カテゴリ: {fact_to_inject['category']}\n- 内容: {fact_to_inject['answer']}"
    user_prompt = f"""
# これまでの会話の流れ
{history_text}
# あなたへのタスク
以下のペルソナ、会話の流れ、そして開示すべき核心的事実のすべてを考慮し、あなた自身の言葉として、この事実を裏付ける具体的なエピソードを含んだ自然な発話を1つだけ生成してください。
# あなたが今回、自然に自己開示するべき「核心的な事実」
{core_fact_to_inject}
# あなたのペルソナ情報（絶対に忘れないでください）
{persona_summary_text}
# 【★★重要ルール★★】
- これは継続した会話です。途中で「こんにちは」のような挨拶を繰り返さないでください。
"""
    return user_prompt

def is_utterance_consistent_with_llm(client, utterance, persona_profile):
    profile_text = "\n".join([f"- {key}: {value}" for key, value in persona_profile.items()])
    judge_prompt = f"""
あなたは、事実の矛盾を厳密にチェックする、高性能な判定AIです。
# ペルソナの確定情報
{profile_text}
# 判定対象の発話
「{utterance}」
# あなたのタスク
上記の「判定対象の発話」が、「ペルソナの確定情報」と明確に矛盾する内容を含んでいるかどうかを判定してください。
判定結果を、必ず以下のJSON形式で出力してください。
{{
  "is_consistent": boolean,
  "reason": "矛盾している、あるいはしていないと判断した簡潔な理由"
}}
"""
    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        judge_result = json.loads(response.choices[0].message.content)
        if not judge_result.get("is_consistent", True):
            print(f"\n⚠️ 矛盾検知(LLM-as-a-judge): {judge_result.get('reason')}")
        return judge_result.get("is_consistent", True)
    except Exception as e:
        print(f"  - 判定APIエラー: {e}")
        return True

def generate_dialogue_for_persona(persona_id):
    print(f"--- ペルソナID: {persona_id} の対話生成を開始します ---")
    client = OpenAI(api_key=Config.API_KEY)
    persona_filepath = os.path.join(Config.PERSONA_DIR, f"persona_{persona_id:02d}.json")
    try:
        with open(persona_filepath, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)
    except FileNotFoundError: return

    facts_to_inject = random.sample(persona_data["other_facts"], Config.NUM_INJECTIONS)
    injection_turns = sorted(random.sample(range(5, Config.NUM_TURNS - 5, 2), Config.NUM_INJECTIONS))
    injection_metadata = []
    dialogue_history = [{"speaker": "assistant", "content": "こんにちは！お元気ですか？"}]
    
    for turn_num in tqdm(range(1, Config.NUM_TURNS + 1), desc=f"ペルソナ {persona_id} 対話生成中"):
        current_role = "user" if turn_num % 2 != 0 else "assistant"
        prompt = ""
        temperature = 0.85
        
        if current_role == "user":
            if turn_num in injection_turns:
                fact_index = injection_turns.index(turn_num)
                fact_to_inject = facts_to_inject[fact_index]
                prompt = create_dynamic_injection_prompt(persona_data, fact_to_inject, dialogue_history)
                temperature = 0.9
                injection_metadata.append({
                    "injection_turn": turn_num, "qa_id": fact_to_inject.get('id', 'N/A'),
                    "category": fact_to_inject.get('category', 'N/A'), "core_fact_answer": fact_to_inject.get('answer', 'N/A')
                })
            else:
                profile_text = "\n".join([f"- {key}: {value}" for key, value in persona_data["persona"]["profile"].items()])
                prompt = f"""直前の相手の発言「{dialogue_history[-1]['content']}」に対し、以下のペルソナとして自然に応答してください。\n# あなたのペルソナ情報（役割）\n{profile_text}\n# 【★★重要ルール★★】\n- これは継続した会話です。途中で「こんにちは」のような挨拶を繰り返さないでください。"""
        else:
            # ★★★ あなたの「会話戦略」ロジックをここに統合 ★★★
            strategy = random.choice(["deepen", "deepen", "connect", "new_topic", "reflect"])
            instruction = ""
            if strategy == "deepen":
                instruction = "ユーザーの直前の発言内容について、さらに一歩踏み込んだオープンな質問を投げかけ、ユーザーの自己開示を促してください。"
            elif strategy == "connect" and len(dialogue_history) > 10:
                user_utterances = [turn['content'] for turn in dialogue_history if turn['speaker'] == 'user']
                if user_utterances:
                    past_utterance = random.choice(user_utterances[:-1])
                    instruction = f"以前ユーザーが「{past_utterance[:30]}...」と話していたことを思い出し、現在の話題と自然に関連付けて会話を広げてください。"
                else:
                    strategy = "reflect" # フォールバック
            elif strategy == "new_topic":
                instruction = "現在の話題が一段落したと判断し、ユーザーが興味を持ちそうな、これまで話していない全く新しい雑談のトピックを自然に提案してください。"
            
            if not instruction or strategy == "reflect":
                instruction = "ユーザーの発言内容を要約し、その内容に対するあなたの理解や共感を示してください。"

            prompt = f"""あなたは聞き上手なAIアシスタントです。ユーザーの発言「{dialogue_history[-1]['content']}」に対して、以下の戦略に従って、親身に、かつ自然に応答してください。\n# 会話戦略\n{instruction}\n# 【★★重要ルール★★】\n- これは継続した会話です。途中で「こんにちは」のような挨拶を繰り返さないでください。\n- 会話を締めくくるような発言（「何か他にありますか？」など）は避け、常に対話が続くようなオープンな応答を心がけてください。"""

        max_retries = 3
        utterance = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=Config.LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, presence_penalty=Config.PRESENCE_PENALTY
                )
                temp_utterance = response.choices[0].message.content.strip()
                if current_role == "user":
                    if is_utterance_consistent_with_llm(client, temp_utterance, persona_data["persona"]["profile"]):
                        utterance = temp_utterance
                        break
                    else:
                        if attempt == max_retries - 1:
                           print(f"\n❌ 再生成リトライ上限到達。")
                           utterance = temp_utterance
                else:
                    utterance = temp_utterance
                    break
            except Exception as e:
                utterance = "(エラーにより発話生成に失敗しました)"
                break
        dialogue_history.append({"speaker": current_role, "content": utterance})
        time.sleep(1)

    if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
    output_filename = os.path.join(Config.OUTPUT_DIR, f"dialogue_p{persona_id:02d}.json")
    with open(output_filename, 'w', encoding='utf-8') as f: json.dump(dialogue_history, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 対話生成が完了し、'{output_filename}' に保存しました。")
    metadata_filename = os.path.join(Config.OUTPUT_DIR, f"dialogue_p{persona_id:02d}.metadata.json")
    with open(metadata_filename, 'w', encoding='utf-8') as f: json.dump(injection_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 注入メタデータを '{metadata_filename}' に保存しました。")


if __name__ == "__main__":
    START_PERSONA_ID = 1
    END_PERSONA_ID = 100 

    print(f"【INFO】ペルソナID {START_PERSONA_ID} から {END_PERSONA_ID} までの大規模対話生成を開始します。")
    for i in range(START_PERSONA_ID, END_PERSONA_ID + 1):
        print("\n" + "="*50)
        generate_dialogue_for_persona(persona_id=i)
        print("="*50)
        if i < END_PERSONA_ID:
            print(f"\nペルソナ {i} の生成完了。次のペルソナに進む前に30秒待機します...")
            time.sleep(30)

    print("\n🎉 全てのペルソナの対話生成が完了しました！")
