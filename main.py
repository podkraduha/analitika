import json
import pandas as pd
import requests

# Использовала API ключ с сайта GroqCloud
GROQ_API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Функция с промтом для неросети
def summarize_news(title, content):
    prompt = f"Сделай краткое содержание новости (1 предложение, максимум 20 слов) на русском:\nЗаголовок: {title}\nТекст: {content}"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        #использую модель llama (нашла ее на сайте groq, из числа доступных)
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Ошибка: {response.status_code}"

def main():
    df = pd.read_csv("news_input.csv")

    results = []
    for idx, row in df.iterrows():
        print(f"Обработка {idx + 1}/{len(df)}: {row['title']}")
        summary = summarize_news(row['title'], row['content'])
        results.append({
            "id": idx,
            "title": row['title'],
            "original_content_preview": row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
            "summary": summary
        })

    with open("output_summaries.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nГотово! Результаты в output_summaries.json")
    # Выводим только первые 100 символов саммари для проверки
    for r in results:
        print(f"{r['title']}: {r['summary'][:100]}...")

if __name__ == "__main__":
    main()