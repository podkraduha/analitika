import json
import pandas as pd
from groq import Groq
#API-ключ с сайта GroqCloud
client = Groq(api_key="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")
#функция с промтом для нейросети
def summarize_news(title, content):
    prompt = f"Сделай краткое содержание новости (1 предложение, максимум 25 слов) на русском. Используй информацию из каждого предложения исходной новости, так чтобы не было похоже на заголовок, но общий его смысл сохранялся:\nЗаголовок: {title}\nТекст: {content}"
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  #доступная и бесплатная модель от Meta
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, #умеренная креативность модели
            max_tokens=300, #длина ответов - максимум 300 токенов
            top_p=1, #параметр разнообразия слов
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Ошибка: {e}" #код ошибки

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

if __name__ == "__main__":
    main()