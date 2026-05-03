"""
Evaluation Charts — رسوم بيانية للنتائج
يُشغّل بعد eval_extraction.py

الاستخدام:
    python -m evaluation.generate_charts
"""
import os, sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def find_latest_results():
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("eval_results_") and f.endswith(".csv")]
    if not csv_files:
        print("❌ لا توجد نتائج! شغّل eval_extraction.py أولاً")
        return None
    return os.path.join(RESULTS_DIR, sorted(csv_files)[-1])


def load_results(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def compute_category_metrics(results):
    expense_results = [r for r in results if r["expected_type"] == "expense" and r["actual_type"] != "error"]
    categories = set(r["expected_category"] for r in expense_results if r["expected_category"])

    metrics = {}
    for cat in categories:
        tp = sum(1 for r in expense_results if r["expected_category"] == cat and r["actual_category"] == cat)
        fp = sum(1 for r in expense_results if r["expected_category"] != cat and r["actual_category"] == cat)
        fn = sum(1 for r in expense_results if r["expected_category"] == cat and r["actual_category"] != cat)

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cat] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return metrics


def chart_1_overall_accuracy(results):
    """دقة النظام الإجمالية"""
    # بدون errors
    no_error = [r for r in results if r["actual_type"] != "error"]
    expense_all = [r for r in results if r["expected_type"] == "expense"]
    expense_clean = [r for r in expense_all if r["actual_type"] != "error"]
    expense_detected = [r for r in expense_all if r["actual_type"] == "expense"]

    type_acc = sum(1 for r in no_error if r["type_correct"] == "True") / len(no_error) * 100 if no_error else 0
    amount_acc = sum(1 for r in expense_detected if r["amount_correct"] == "True") / len(expense_detected) * 100 if expense_detected else 0
    cat_acc = sum(1 for r in expense_detected if r["category_correct"] == "True") / len(expense_detected) * 100 if expense_detected else 0
    full_acc = sum(1 for r in expense_clean if r["status"] == "✅") / len(expense_clean) * 100 if expense_clean else 0

    labels = ["Type\nClassification", "Amount\nExtraction", "Category\nClassification", "Full\nExtraction"]
    values = [type_acc, amount_acc, cat_acc, full_acc]
    colors = ["#2B6CB0", "#48BB78", "#ED8936", "#E53E3E"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Overall System Accuracy (excluding timeout errors)", fontsize=15, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=85, color="red", linestyle="--", alpha=0.5, label="Target: 85%")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "1_overall_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 1_overall_accuracy.png")


def chart_2_f1_per_category(cat_metrics):
    """F1-Score لكل فئة"""
    sorted_cats = sorted(cat_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    labels = [c[0] for c in sorted_cats]
    f1_scores = [c[1]["f1"] for c in sorted_cats]
    colors = ["#48BB78" if f >= 85 else "#ED8936" if f >= 70 else "#E53E3E" for f in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, f1_scores, color=colors, height=0.6, edgecolor="white")

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("F1-Score (%)", fontsize=13)
    ax.set_title("F1-Score per Category", fontsize=15, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=83.5, color="red", linestyle="--", alpha=0.5, label="Target: 83.5%")
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "2_f1_per_category.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 2_f1_per_category.png")


def chart_3_precision_recall_f1(cat_metrics):
    """Precision vs Recall vs F1"""
    sorted_cats = sorted(cat_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    labels = [c[0] for c in sorted_cats]
    precision = [c[1]["precision"] for c in sorted_cats]
    recall = [c[1]["recall"] for c in sorted_cats]
    f1 = [c[1]["f1"] for c in sorted_cats]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, precision, width, label="Precision", color="#2B6CB0", edgecolor="white")
    ax.bar(x, recall, width, label="Recall", color="#48BB78", edgecolor="white")
    ax.bar(x + width, f1, width, label="F1-Score", color="#ED8936", edgecolor="white")

    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Precision, Recall & F1-Score per Category", fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "3_precision_recall_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 3_precision_recall_f1.png")


def chart_4_results_distribution(results):
    """توزيع الصح والغلط"""
    expense_results = [r for r in results if r["expected_type"] == "expense"]
    correct = sum(1 for r in expense_results if r["status"] == "✅")
    wrong_classify = sum(1 for r in expense_results if r["status"] == "❌" and r["actual_type"] != "error")
    timeout = sum(1 for r in expense_results if r["actual_type"] == "error")

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [correct, wrong_classify, timeout]
    labels = [f"Correct\n({correct})", f"Wrong Category\n({wrong_classify})", f"Timeout/Error\n({timeout})"]
    colors = ["#48BB78", "#ED8936", "#E53E3E"]
    explode = (0.05, 0.05, 0.05)

    # حذف الصفر
    filtered = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
    if filtered:
        sizes, labels, colors, explode = zip(*filtered)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Expense Extraction Results", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "4_results_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 4_results_distribution.png")


def chart_5_type_classification(results):
    """دقة تحديد نوع الرسالة"""
    no_error = [r for r in results if r["actual_type"] != "error"]
    types = {"expense": [], "query": [], "advice": [], "chat": []}

    for r in no_error:
        et = r["expected_type"]
        if et in types:
            types[et].append(r["type_correct"] == "True")
        elif et == "not_expense":
            types["chat"].append(r["type_correct"] == "True")

    labels = []
    accuracies = []
    counts = []
    for t, vals in types.items():
        if vals:
            labels.append(t.capitalize())
            accuracies.append(sum(vals) / len(vals) * 100)
            counts.append(len(vals))

    colors = ["#2B6CB0", "#48BB78", "#ED8936", "#805AD5"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accuracies, color=colors[:len(labels)], width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val, cnt in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Message Type Classification Accuracy", fontsize=15, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "5_type_classification.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 5_type_classification.png")


def chart_6_comparison_with_studies(cat_metrics):
    """مقارنة مع الدراسات السابقة"""
    macro_f1 = sum(m["f1"] for m in cat_metrics.values()) / len(cat_metrics) if cat_metrics else 0

    studies = [
        ("Study 2\n(NAACL 2024)", 60),
        ("Study 5\n(LLM-RAG)", 78.6),
        ("Study 3\n(HierFinRAG)", 83.5),
        ("Our System\n(Mizan)", macro_f1),
    ]

    labels = [s[0] for s in studies]
    values = [s[1] for s in studies]
    colors = ["#A0AEC0", "#A0AEC0", "#A0AEC0", "#2B6CB0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("F1-Score / Accuracy (%)", fontsize=13)
    ax.set_title("Comparison with Previous Studies", fontsize=15, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "6_comparison_studies.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 6_comparison_studies.png")


def main():
    print("=" * 60)
    print("📊 توليد الرسوم البيانية...")
    print("=" * 60)

    results_file = find_latest_results()
    if not results_file:
        return

    print(f"\n📂 تحميل النتائج من: {results_file}")
    results = load_results(results_file)
    print(f"   تم تحميل {len(results)} نتيجة")

    cat_metrics = compute_category_metrics(results)

    print(f"\n📈 توليد الرسوم في: {CHARTS_DIR}")
    chart_1_overall_accuracy(results)
    chart_2_f1_per_category(cat_metrics)
    chart_3_precision_recall_f1(cat_metrics)
    chart_4_results_distribution(results)
    chart_5_type_classification(results)
    chart_6_comparison_with_studies(cat_metrics)

    print(f"\n✅ تم توليد 6 رسوم بيانية!")
    print(f"   📁 المسار: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
