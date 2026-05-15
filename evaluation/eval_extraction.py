"""
Evaluation Script — تقييم دقة النظام
مع retry محسّن وتأخير لتجنب rate limiting

الاستخدام:
    $env:GROQ_API_KEY="gsk_xxx"
    python -m evaluation.eval_extraction
"""
import os, sys, csv, json, time
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm.parser import ExpenseParser
from config import GROQ_API_KEY

TEST_FILE = os.path.join(os.path.dirname(__file__), "test_dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "sentence": row["sentence"].strip(),
                "expected_amount": float(row["expected_amount"]) if row["expected_amount"] else None,
                "expected_category": row["expected_category"].strip() if row["expected_category"] else None,
                "expected_type": row["expected_type"].strip(),
            })
    return data


def evaluate(parser, test_data):
    results = []
    total = len(test_data)
    errors_count = 0

    print(f"\n🧪 تقييم {total} جملة (مع retry تلقائي)...")
    print("=" * 60)

    for i, item in enumerate(test_data):
        sentence = item["sentence"]
        expected_type = item["expected_type"]
        expected_amount = item["expected_amount"]
        expected_category = item["expected_category"]

        # تأخير بين كل جملة لتجنب rate limiting
        if i > 0:
            time.sleep(4)
        if i > 0 and i % 10 == 0:
            print(f"   ... تم {i}/{total} (أخطاء: {errors_count})")

        # استدعاء الـ Parser
        try:
            result = parser.parse_expense(sentence)
            actual_type = result.get("type", "error")

            # إذا طلع error، جرّب مرة ثانية بعد انتظار أطول
            if actual_type == "error":
                time.sleep(10)
                result = parser.parse_expense(sentence)
                actual_type = result.get("type", "error")

            # تحقق من نوع الرسالة
            type_match = False
            if expected_type in ["chat", "not_expense"] and actual_type in ["chat", "not_expense"]:
                type_match = True
            elif expected_type == actual_type:
                type_match = True

            # تحقق من المبلغ والفئة
            amount_match = False
            category_match = False

            if expected_type == "expense" and actual_type == "expense":
                actual_amount = result.get("amount")
                actual_category = result.get("category", "")

                if actual_amount and expected_amount:
                    diff = abs(actual_amount - expected_amount) / expected_amount
                    amount_match = diff < 0.05

                category_match = actual_category == expected_category

            is_correct = type_match and (expected_type != "expense" or (amount_match and category_match))
            if not is_correct:
                errors_count += 1

            results.append({
                "sentence": sentence,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "type_correct": type_match,
                "expected_amount": expected_amount,
                "actual_amount": result.get("amount") if actual_type == "expense" else None,
                "amount_correct": amount_match,
                "expected_category": expected_category,
                "actual_category": result.get("category", "") if actual_type == "expense" else None,
                "category_correct": category_match,
                "status": "✅" if is_correct else "❌"
            })

        except Exception as e:
            errors_count += 1
            results.append({
                "sentence": sentence,
                "expected_type": expected_type,
                "actual_type": "error",
                "type_correct": False,
                "expected_amount": expected_amount,
                "actual_amount": None,
                "amount_correct": False,
                "expected_category": expected_category,
                "actual_category": None,
                "category_correct": False,
                "status": "❌",
                "error": str(e)
            })

    print(f"   ... تم {total}/{total}")
    return results


def compute_metrics(results):
    metrics = {}

    # 1. دقة تحديد النوع
    type_correct = sum(1 for r in results if r["type_correct"])
    metrics["type_accuracy"] = type_correct / len(results) * 100

    # 2. فلترة الأخطاء التقنية (timeout/error)
    non_error_results = [r for r in results if r["actual_type"] != "error"]
    error_count = sum(1 for r in results if r["actual_type"] == "error")
    metrics["error_count"] = error_count
    metrics["error_rate"] = error_count / len(results) * 100

    # 3. دقة المصاريف (بدون الأخطاء التقنية)
    expense_results = [r for r in results if r["expected_type"] == "expense"]
    expense_no_error = [r for r in expense_results if r["actual_type"] != "error"]
    expense_detected = [r for r in expense_results if r["actual_type"] == "expense"]

    if expense_results:
        full_correct = sum(1 for r in expense_results if r["type_correct"] and r["amount_correct"] and r["category_correct"])
        metrics["extraction_accuracy_all"] = full_correct / len(expense_results) * 100

    if expense_no_error:
        full_correct_no_err = sum(1 for r in expense_no_error if r["type_correct"] and r["amount_correct"] and r["category_correct"])
        metrics["extraction_accuracy_clean"] = full_correct_no_err / len(expense_no_error) * 100

    if expense_detected:
        amount_correct = sum(1 for r in expense_detected if r["amount_correct"])
        metrics["amount_accuracy"] = amount_correct / len(expense_detected) * 100

        cat_correct = sum(1 for r in expense_detected if r["category_correct"])
        metrics["category_accuracy"] = cat_correct / len(expense_detected) * 100

    # 4. F1-Score لكل فئة (بدون errors)
    categories = set(r["expected_category"] for r in expense_no_error if r["expected_category"])
    category_metrics = {}

    for cat in categories:
        tp = sum(1 for r in expense_no_error if r["expected_category"] == cat and r["actual_category"] == cat)
        fp = sum(1 for r in expense_no_error if r["expected_category"] != cat and r["actual_category"] == cat)
        fn = sum(1 for r in expense_no_error if r["expected_category"] == cat and r["actual_category"] != cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        category_metrics[cat] = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "tp": tp, "fp": fp, "fn": fn
        }

    metrics["category_f1"] = category_metrics

    # 5. Macro averages
    if category_metrics:
        metrics["macro_precision"] = sum(m["precision"] for m in category_metrics.values()) / len(category_metrics)
        metrics["macro_recall"] = sum(m["recall"] for m in category_metrics.values()) / len(category_metrics)
        metrics["macro_f1"] = sum(m["f1"] for m in category_metrics.values()) / len(category_metrics)

    # 6. دقة الاستعلامات
    query_results = [r for r in results if r["expected_type"] in ["query", "advice", "chat", "not_expense"]]
    if query_results:
        query_correct = sum(1 for r in query_results if r["type_correct"])
        metrics["query_accuracy"] = query_correct / len(query_results) * 100

    return metrics


def print_results(results, metrics):
    print("\n" + "=" * 60)
    print("📊 نتائج التقييم")
    print("=" * 60)

    print(f"\n📋 عدد الجمل: {len(results)}")
    print(f"   ✅ صحيحة: {sum(1 for r in results if r['status'] == '✅')}")
    print(f"   ❌ خاطئة: {sum(1 for r in results if r['status'] == '❌')}")
    print(f"   ⚠️ أخطاء تقنية (timeout): {metrics.get('error_count', 0)} ({metrics.get('error_rate', 0):.1f}%)")

    print(f"\n{'─' * 60}")
    print("🎯 المقاييس الرئيسية:")
    print(f"{'─' * 60}")
    print(f"   دقة تحديد النوع:            {metrics.get('type_accuracy', 0):.1f}%")
    print(f"   دقة الاستخراج (مع errors):   {metrics.get('extraction_accuracy_all', 0):.1f}%")
    print(f"   دقة الاستخراج (بدون errors): {metrics.get('extraction_accuracy_clean', 0):.1f}%")
    print(f"   دقة المبلغ:                  {metrics.get('amount_accuracy', 0):.1f}%")
    print(f"   دقة الفئة:                   {metrics.get('category_accuracy', 0):.1f}%")
    print(f"   دقة الاستعلامات:             {metrics.get('query_accuracy', 0):.1f}%")

    print(f"\n{'─' * 60}")
    print(f"   📊 Macro Precision:           {metrics.get('macro_precision', 0):.1f}%")
    print(f"   📊 Macro Recall:              {metrics.get('macro_recall', 0):.1f}%")
    print(f"   📊 Macro F1-Score:            {metrics.get('macro_f1', 0):.1f}%")
    print(f"{'─' * 60}")

    print(f"\n📂 F1-Score حسب الفئة:")
    print(f"{'─' * 60}")
    print(f"   {'الفئة':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   {'─' * 52}")

    cat_f1 = metrics.get("category_f1", {})
    for cat, m in sorted(cat_f1.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"   {cat:<20} {m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}%")

    print(f"{'─' * 60}")

    # الأخطاء (بدون timeout)
    real_errors = [r for r in results if r["status"] == "❌" and r["actual_type"] != "error"]
    timeout_errors = [r for r in results if r["actual_type"] == "error"]

    if real_errors:
        print(f"\n❌ أخطاء تصنيف ({len(real_errors)}):")
        print(f"{'─' * 60}")
        for e in real_errors[:15]:
            print(f"   \"{e['sentence']}\"")
            print(f"     متوقع: {e['expected_category']} | فعلي: {e.get('actual_category', '')}")
            print()

    if timeout_errors:
        print(f"\n⚠️ أخطاء تقنية/timeout ({len(timeout_errors)}):")
        for e in timeout_errors[:5]:
            print(f"   \"{e['sentence']}\"")


def save_results(results, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(RESULTS_DIR, f"eval_results_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentence", "expected_type", "actual_type", "type_correct",
            "expected_amount", "actual_amount", "amount_correct",
            "expected_category", "actual_category", "category_correct", "status"
        ])
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k in writer.fieldnames}
            writer.writerow(row)
    print(f"\n💾 نتائج تفصيلية: {csv_path}")

    summary_path = os.path.join(RESULTS_DIR, f"eval_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Smart Family Budget Assistant — Evaluation Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total sentences: {len(results)}\n")
        f.write(f"Technical errors (timeout): {metrics.get('error_count', 0)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Main Metrics (excluding timeout errors):\n")
        f.write(f"  Type Accuracy:            {metrics.get('type_accuracy', 0):.1f}%\n")
        f.write(f"  Extraction Accuracy:      {metrics.get('extraction_accuracy_clean', 0):.1f}%\n")
        f.write(f"  Amount Accuracy:          {metrics.get('amount_accuracy', 0):.1f}%\n")
        f.write(f"  Category Accuracy:        {metrics.get('category_accuracy', 0):.1f}%\n")
        f.write(f"  Query Accuracy:           {metrics.get('query_accuracy', 0):.1f}%\n\n")

        f.write(f"  Macro Precision:          {metrics.get('macro_precision', 0):.1f}%\n")
        f.write(f"  Macro Recall:             {metrics.get('macro_recall', 0):.1f}%\n")
        f.write(f"  Macro F1-Score:           {metrics.get('macro_f1', 0):.1f}%\n\n")

        f.write("F1-Score per Category:\n")
        cat_f1 = metrics.get("category_f1", {})
        for cat, m in sorted(cat_f1.items(), key=lambda x: x[1]["f1"], reverse=True):
            f.write(f"  {cat}: P={m['precision']:.1f}% R={m['recall']:.1f}% F1={m['f1']:.1f}%\n")

    print(f"💾 ملخص التقييم: {summary_path}")
    return csv_path, summary_path


def main():
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️ مطلوب مفتاح Groq API!")
        print('  PowerShell: $env:GROQ_API_KEY="gsk_xxx"')
        return

    parser = ExpenseParser(api_key=api_key)

    print("🔌 اختبار الاتصال بـ Groq...")
    if not parser.test_connection():
        return

    print(f"\n📂 تحميل بيانات الاختبار...")
    test_data = load_test_data(TEST_FILE)
    print(f"   تم تحميل {len(test_data)} جملة")

    expenses = [d for d in test_data if d["expected_type"] == "expense"]
    queries = [d for d in test_data if d["expected_type"] == "query"]
    advice = [d for d in test_data if d["expected_type"] == "advice"]
    chat = [d for d in test_data if d["expected_type"] == "chat"]
    print(f"   مصاريف: {len(expenses)} | استعلامات: {len(queries)} | نصائح: {len(advice)} | محادثة: {len(chat)}")

    start_time = time.time()
    results = evaluate(parser, test_data)
    elapsed = time.time() - start_time

    metrics = compute_metrics(results)
    metrics["total_time"] = elapsed
    metrics["avg_time"] = elapsed / len(results)

    print(f"\n⏱️ الوقت: {elapsed:.0f} ثانية ({metrics['avg_time']:.1f} ثانية/جملة)")

    print_results(results, metrics)
    save_results(results, metrics)

    print("\n✅ التقييم انتهى!")
    print(f"\n💡 الخطوة التالية: python -m evaluation.generate_charts")


if __name__ == "__main__":
    main()
