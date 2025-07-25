# merge_batches.py

import json
import glob
import os
import argparse

def main(input_dir, output_file):
    """
    指定されたディレクトリ内のバッチファイル（batch_XXX.json）を結合し、
    最終的なIDを1から振り直して、単一のJSONファイルとして出力する。
    """
    # バッチ処理で生成されるファイル名パターン
    file_pattern = "batch_*.json"
    search_path = os.path.join(input_dir, file_pattern)
    
    # globの結果は順序が保証されないため、ファイル名でソートして処理の順序を安定させる
    batch_files = sorted(glob.glob(search_path))

    if not batch_files:
        print(f"エラー: ディレクトリ '{input_dir}' に結合対象のバッチファイル（{file_pattern}）が見つかりません。")
        return

    print(f"{len(batch_files)}個のバッチファイルを結合します...")
    print("-" * 20)

    all_data = []
    for filepath in batch_files:
        print(f"  > 読み込み中: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except json.JSONDecodeError:
                print(f"    - 警告: {os.path.basename(filepath)} は不正なJSONファイルです。スキップします。")

    # 【重要】最終的なIDを1から振り直す
    print("-" * 20)
    print("全データを結合し、最終IDを振り直しています...")
    for i, item in enumerate(all_data):
        item['id'] = i + 1

    # 最終的なファイルに書き出す
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 結合とID付与が完了しました。")
    print(f"   - 出力ファイル: {output_file}")
    print(f"   - 合計件数: {len(all_data)}件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAペアのバッチファイルを結合し、最終IDを付与するスクリプト")
    
    # Colabでもローカルでも使いやすいように、引数でディレクトリとファイル名を指定可能にする
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='batches', 
        help='バッチファイルが格納されているディレクトリのパス'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='QA_pairs_5000_final.json', 
        help='最終的な出力ファイル名'
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_file)
