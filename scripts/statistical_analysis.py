import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

# =====================================================
# User config: replace with your local paths before run
# =====================================================
MODELS_CONFIG = {
    # Full system
    'FoodAudit-AG (Full System)': r'E:\Chain of Thought in llm\database_analysis\test_final_report.xlsx',

    # Baselines
    'Keyword-Hierarchy': r'E:\Chain of Thought in llm\scripts_test\baseline_reports\keyword_hierarchy.xlsx',
    'Keyword-NoHier': r'E:\Chain of Thought in llm\scripts_test\baseline_reports\keyword_nohier.xlsx',
    'LLM-Only': r'E:\Chain of Thought in llm\scripts_test\baseline_reports\llm_only.xlsx',
    'Hierarchy-exposed Vector-RAG': r'E:\Chain of Thought in llm\scripts_test\baseline_reports\vector_rag.xlsx',
    'Corrective Vector-RAG': r'E:\Chain of Thought in llm\scripts_test\baseline_reports_new\corrective_vector_rag.xlsx',
    'Logic-aware Hierarchy-RAG': r'E:\Chain of Thought in llm\scripts_test\baseline_reports_new\logic_aware_hierarchy_rag.xlsx',

    # Ablations
    'No Bi-Retrieval': r'E:\Chain of Thought in llm\scripts_test\ablation_reports\bi_retrieval_off.xlsx',
    'No Hierarchy': r'E:\Chain of Thought in llm\scripts_test\ablation_reports\hierarchy_off.xlsx',
    'No Whitelist': r'E:\Chain of Thought in llm\scripts_test\ablation_reports\whitelist_negation_off.xlsx',
}

REFERENCE_MODEL_NAME = 'FoodAudit-AG (Full System)'
BOOTSTRAP_N = 1000
SEED = 42

DISPLAY_ORDER = [
    'FoodAudit-AG (Full System)',
    'Hierarchy-exposed Vector-RAG',
    'Logic-aware Hierarchy-RAG',
    'Corrective Vector-RAG',
    'Keyword-Hierarchy',
    'Keyword-NoHier',
    'LLM-Only',
    'No Bi-Retrieval',
    'No Hierarchy',
    'No Whitelist',
]


def load_data_robustly(filepath: str):
    if not os.path.exists(filepath):
        print(f'⚠️ 找不到文件，跳过: {filepath}')
        return None

    if filepath.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        df = None
        for enc in ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']:
            try:
                df = pd.read_csv(filepath, encoding=enc, engine='python')
                break
            except Exception:
                continue
        if df is None:
            raise ValueError(f'无法读取文件: {filepath}')

    # column normalization
    rename_map = {}
    if 'gt_map' in df.columns and 'ground_truth' not in df.columns:
        rename_map['gt_map'] = 'ground_truth'
    if 'difficulty_tier' in df.columns and 'difficulty_level' not in df.columns:
        rename_map['difficulty_tier'] = 'difficulty_level'
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ['fine_correct', 'binary_correct']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'{filepath} 缺少必要列: {col}')

    df['fine_correct'] = pd.to_numeric(df['fine_correct'], errors='coerce').fillna(0).astype(int)
    df['binary_correct'] = pd.to_numeric(df['binary_correct'], errors='coerce').fillna(0).astype(int)

    if 'latency' in df.columns:
        df['latency'] = pd.to_numeric(df['latency'], errors='coerce')

    if 'ground_truth' in df.columns:
        df['ground_truth'] = df['ground_truth'].fillna('').astype(str)
    else:
        df['ground_truth'] = ''

    return df


def calculate_bootstrap_ci(values, n_iterations=BOOTSTRAP_N, seed=SEED):
    np.random.seed(seed)
    values = np.asarray(values, dtype=float)
    n_size = len(values)
    scores = [np.random.choice(values, size=n_size, replace=True).mean() for _ in range(n_iterations)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def normalize_diff_name(x: str) -> str:
    s = str(x).strip().upper()
    if s.startswith('L1'):
        return 'L1'
    if s.startswith('L2'):
        return 'L2'
    if s.startswith('L3'):
        return 'L3'
    if s.startswith('L4'):
        return 'L4'
    return s


def format_pvalue(p: float) -> str:
    if pd.isna(p):
        return '-'
    if p < 0.001:
        return '< 0.001'
    return f'{p:.4f}'


def calculate_mcnemar(df_ref: pd.DataFrame, df_target: pd.DataFrame):
    if df_ref is None or df_target is None or 'id' not in df_ref.columns or 'id' not in df_target.columns:
        return None, None

    merged = pd.merge(
        df_ref[['id', 'fine_correct']],
        df_target[['id', 'fine_correct']],
        on='id',
        suffixes=('_ref', '_tgt')
    )
    if merged.empty:
        return None, None

    n11 = int(((merged['fine_correct_ref'] == 1) & (merged['fine_correct_tgt'] == 1)).sum())
    n10 = int(((merged['fine_correct_ref'] == 1) & (merged['fine_correct_tgt'] == 0)).sum())
    n01 = int(((merged['fine_correct_ref'] == 0) & (merged['fine_correct_tgt'] == 1)).sum())
    n00 = int(((merged['fine_correct_ref'] == 0) & (merged['fine_correct_tgt'] == 0)).sum())

    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.pvalue), table


def compute_binary_prf(df: pd.DataFrame):
    y_true = df['ground_truth'].apply(lambda x: 1 if 'RISK' in str(x).upper() else 0).to_numpy()
    y_pred = np.where(df['binary_correct'].to_numpy() == 1, y_true, 1 - y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(precision), float(recall), float(f1)


def main():
    print('🚀 开始汇总全部模型结果...')
    data_dict = {}
    for model_name, path in MODELS_CONFIG.items():
        df = load_data_robustly(path)
        if df is not None:
            data_dict[model_name] = df
            print(f'✅ {model_name}: n={len(df)}')

    if not data_dict:
        print('❌ 没有可用数据。')
        return

    ref_df = data_dict.get(REFERENCE_MODEL_NAME)
    rows = []
    pvalue_rows = []

    for model_name, df in data_dict.items():
        overall_acc = float(df['fine_correct'].mean())
        ci_lower, ci_upper = calculate_bootstrap_ci(df['fine_correct'].values)

        diff_series = df.assign(difficulty_norm=df['difficulty_level'].apply(normalize_diff_name)) \
                        .groupby('difficulty_norm')['fine_correct'].mean()
        l1 = float(diff_series.get('L1', np.nan))
        l2 = float(diff_series.get('L2', np.nan))
        l3 = float(diff_series.get('L3', np.nan))
        l4 = float(diff_series.get('L4', np.nan))

        precision, recall, f1 = compute_binary_prf(df)

        lat_mean = float(df['latency'].mean()) if 'latency' in df.columns else np.nan
        lat_std = float(df['latency'].std()) if 'latency' in df.columns else np.nan

        if model_name == REFERENCE_MODEL_NAME:
            p_raw, table = np.nan, None
        else:
            p_raw, table = calculate_mcnemar(ref_df, df)

        rows.append({
            'Model': model_name,
            'Accuracy': overall_acc,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Overall Accuracy [95% CI]': f'{overall_acc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]',
            'L1 Acc': l1,
            'L2 Acc': l2,
            'L3 Acc': l3,
            'L4 Acc': l4,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Latency Mean (s)': lat_mean,
            'Latency Std (s)': lat_std,
            'Latency (s)': '-' if pd.isna(lat_mean) else f'{lat_mean:.2f} ± {lat_std:.2f}',
            'p-value raw': p_raw,
            'p-value (vs Full System)': '-' if model_name == REFERENCE_MODEL_NAME else format_pvalue(p_raw),
        })

        if table is not None:
            pvalue_rows.append({
                'Model': model_name,
                'n11_both_correct': table[0][0],
                'n10_ref_only_correct': table[0][1],
                'n01_target_only_correct': table[1][0],
                'n00_both_wrong': table[1][1],
                'p_value': p_raw,
                'p_value_display': format_pvalue(p_raw),
            })

    final_df = pd.DataFrame(rows)
    final_df['order'] = final_df['Model'].apply(lambda x: DISPLAY_ORDER.index(x) if x in DISPLAY_ORDER else 999)
    final_df = final_df.sort_values(['order', 'Model']).drop(columns='order')

    final_df.to_csv('master_table_for_paper_updated.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pvalue_rows).to_csv('mcnemar_details_updated.csv', index=False, encoding='utf-8-sig')

    print('\n🎉 完成：')
    print(' - master_table_for_paper_updated.csv')
    print(' - mcnemar_details_updated.csv')
    print('\n提示：只要 corrective_vector_rag.xlsx 和 logic_aware_hierarchy_rag.xlsx 里有 id 与 fine_correct，本脚本会自动计算它们的 McNemar p-value。')


if __name__ == '__main__':
    main()
