#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.sparse import coo_array, csc_array, coo_matrix, csc_matrix
import pandas as pd
import itertools
import re
import matplotlib.pyplot as plt
import sys
from highspy import Highs, HighsStatus, ObjSense, HighsModelStatus

def confirm(prompt="続行しますか？ [y/N]: ", default=False):
    """
    y/yesでTrue、n/noでFalse。未入力はdefault。
    大文字・小文字・全角y/ｙにも対応。
    """
    try:
        s = input(prompt).strip().lower()
    except EOFError:
        # パイプや非対話環境などでstdinがない場合
        return default

    if s in {"y", "yes", "ｙ"}:
        return True
    if s in {"n", "no", "ｎ"}:
        return False
    # 未入力/その他は既定値
    return default

def generate_mps_file(var_num, var_list, A_csc, b, c):
    replace_var_name = lambda x:x.replace(' ', '').replace(',','_').replace('(', '').replace(')','')

    coo = A_csc.tocoo()
    rows = coo.row.tolist()
    cols = coo.col.tolist()
    vals = coo.data.tolist()

    begin_int = var_list['s'].__len__() + var_list['t'].__len__()
    print('begin_int:', begin_int)

    with open('taskschedule.mps', 'w') as f:
        print('NAME TASKS', file=f)
        print('ROWS', file=f)
        print(' N OBJ', file=f)
        for i in range(len(b)):
            print(' L c{}'.format(i), file=f)

        print('COLUMNS', file=f)
        pre_v = 0
        int_flg = False
        for i in range(len(rows)):
            if pre_v != cols[i]:
                print(' v{} OBJ {:.5f}'.format(cols[i-1], c[cols[i-1]]), file=f)
                pre_v = cols[i]
            if cols[i] == begin_int and int_flg == False:
                print(" MARKER 'MARKER' 'INTORG'", file=f)
                int_flg = True
            print(' v{} c{} {:.5f}'.format(cols[i], rows[i], vals[i]), file=f)
        print(' v{} OBJ {:.5f}'.format(cols[i], c[cols[i]]), file=f)
        print(" MARKER 'MARKER' 'INTEND'", file=f)

        print('RHS', file=f)
        for i in range(len(b)):
            print(' RHS1 c{} {:.5f}'.format(i, b[i]), file=f)
        
        print('BOUNDS', file=f)
        pre_v = 0
        int_flg = False
        for i in range(var_num):
            if i < begin_int:
                print(' UP BND1 v{} {:.5f}'.format(i, var_list['M']), file=f)
            else:
                print(' UP BND1 v{} {:.5f}'.format(i, 1), file=f)
            print(' LO BND1 v{} {:.5f}'.format(i, 0), file=f)
        print('ENDATA', file=f)
    return

# 変数リスト生成
# s: タスクiのn回目の開始時間
# t: タスクiのn回目の実行のデッドラインからの遅れ
# y: タスクiのn回目がプロセッサpで実行されるか否か
# x: タスクiのn回目とタスクjのm回目が同一プロセッサで実行される場合の実行順序に制約が存在するか否か
# n_range: タスクごとの実行回数
# 割り当て禁止プロセッサに対する変数y(i, n, p)を生成しない
# 同一プロセッサに対する割り当てが発生しない場合は変数x((i, n), (j, m))を生成しない
def make_variable_list(df, processor_num=1, Max_Period=None):
    HyperPeriod = np.lcm.reduce(df['周期T(i)'].values)
    if Max_Period is not None:
        HyperPeriod = min(Max_Period, HyperPeriod)
    #print(HyperPeriod)
    pairs = [(i['ID'], j) for _, i in df.iterrows() for j in  list(range(1, int(HyperPeriod / i['周期T(i)']) + 1))]
    n_range = [range(1, int(HyperPeriod / i['周期T(i)']) + 1) for _, i in df.iterrows()]
    #print(pairs)

    processor_list = [i for i in range(processor_num)]

    # タスクiに対する割り当てプロセッサリスト
    processor_list_for_task = {}
    for id, _ in pairs:
        if df.at[id, '専用プロセッサリスト'] == '-':
            processor_list_for_task[id] = processor_list
        else:
            tmp = df.at[id, '専用プロセッサリスト']
            if type(tmp) == np.int64:
                processor_list_for_task[id] = [int(tmp)]
            else:
                processor_list_for_task[id] = list(map(int, tmp.split(',')))
    #print(processor_list_for_task)

    # 同じプロセッサで実行しない排他タスクリスト
    exclude_task_pairs = []
    for i in df['ID'].values:
        for j in df['ID'].values:
            if i == j:
                continue
            if set(processor_list_for_task[i]) & set(processor_list_for_task[j]) == set():
                exclude_task_pairs.append((int(i), int(j)))

    #print(exclude_task_pairs)

    s = ['s{}'.format(i) for i in pairs]
    t = ['t{}'.format(i) for i in pairs]
    tmp = [(a, b) for a, b in itertools.product(pairs, pairs) if a != b]
    x = ['x{}'.format(i) for i in tmp if (i[0][0], i[1][0]) not in exclude_task_pairs]
    y = ['y{}'.format(i) for i in itertools.product(pairs, processor_list) if i[1] in processor_list_for_task[i[0][0]]]
    return {'M':HyperPeriod, 's':s, 't':t, 'x':x, 'y':y, 'n_range':n_range, 'P':processor_list}

def get_s_index(variable_list, id, n):
    try:
        return variable_list['s'].index('s({}, {})'.format(id, n))
    except ValueError:
        return -1

def get_t_index(variable_list, id, n):
    try:
        return variable_list['t'].index('t({}, {})'.format(id, n)) + len(variable_list['s'])
    except ValueError:
        return -1

def get_x_index(variable_list, id1, n1, id2, n2):
    try:
        return variable_list['x'].index('x(({}, {}), ({}, {}))'.format(id1, n1, id2, n2)) + len(variable_list['s']) + len(variable_list['t'])
    except ValueError:
        return -1

def get_x_index_from_str(variable_list, x):
    try:
        return variable_list['x'].index(x) + len(variable_list['s']) + len(variable_list['t'])
    except ValueError:
        return -1

def get_index_from_x(x):
    pattern = re.compile(
        r"""^x\(\(                 # 先頭 'x(('
            \s*(?P<ID1>-?\d+)\s*,\s*(?P<n1>-?\d+)\s*   # 1組目 
            \)\s*,\s*\(                                 # '),(' の区切り
            \s*(?P<ID2>-?\d+)\s*,\s*(?P<n2>-?\d+)\s*   # 2組目 
            \)\s*\)$                                    # 末尾 '))'
        """,
        re.VERBOSE
    )

    return pattern.match(x).groupdict()


def get_y_index(variable_list, id, n, p):
    try:
        return variable_list['y'].index('y(({}, {}), {})'.format(id, n, p))  + len(variable_list['s']) + len(variable_list['t']) + len(variable_list['x'])
    except ValueError:
        return -1

def boundaries(a):
    b = [0]  # 先頭は0
    for i in range(1, len(a)):
        if a[i] != a[i-1]:
            b.append(i)
    b.append(len(a))  # 終端境界（系列長）
    return b

# 式生成
def solve_MILP_csc(df, processor_num=2, verbose=False):
    # 定数設定
    # タスクID
    IDs = df['ID'].values.tolist()
    # 周期
    T = df['周期T(i)'].values.tolist()
    # 実行時間
    E = df['実行時間E(i)'].values.tolist()
    # デッドライン
    D = df['デッドラインD(i)'].values.tolist()
    # 占有プロセッサ数
    NumP = df['使用プロセッサ数P(i)'].values.tolist()

    if processor_num < max(NumP):
        print('warning: processor_num < max(NumP)')
        processor_num = max(NumP)

    # プロセッサ数
    Ps = [i for i in range(processor_num)]

    # 変数生成
    #variable_list = make_variable_list(df=df, processor_num=processor_num, Max_Period=16)
    variable_list = make_variable_list(df=df, processor_num=processor_num)
    variable_num = len(variable_list['s']) + len(variable_list['t']) + len(variable_list['y']) + len(variable_list['x'])
    print('variable_num: {}, Hyperperiod: {}'.format(variable_num, variable_list['M']))
    # big-M 設定
    M = variable_list['M']

    # 最適化条件設定
    c = np.zeros(variable_num, dtype=float)
    # tの和を最小にする
    c[len(variable_list['s']):len(variable_list['s']) + len(variable_list['t'])] = 1
    # print(c)

    # 制約条件設定
    A_list = []
    A_cols = []
    A_rows = []
    A_row = 0
    A_vals = []
    b_list = []
    label_list = []
    for id in IDs:
        for n in variable_list['n_range'][id]:
            # print('ID:', id, 'n:', n)
            # n > 0
            # ① デッドライン制約：s(i, n) - t(i, n) <= (n-1) * T(i) + D(i)- E(i)
            A_cols.extend([get_s_index(variable_list, id, n), get_t_index(variable_list, id, n)])
            A_rows.extend([A_row, A_row])
            A_row = A_row + 1
            A_vals.extend([1, -1])
            b_list.append((n - 1) * T[id] + D[id] - E[id])
            label_list.append("①")

            # ② 周期制約：s(i, n) - s(i, n + 1) <= - T(i)
            # 最後の実行の場合はスキップ
            if n < len(variable_list['n_range'][id]):
                A_cols.extend([get_s_index(variable_list, id, n), get_s_index(variable_list, id, n + 1)])
                A_rows.extend([A_row, A_row])
                A_row = A_row + 1
                A_vals.extend([1, -1])
                b_list.append(-T[id])
                label_list.append("②")


            # ③ データ依存制約：s(j, n) - s(i, n) <= -E(j)
            if df.at[id, '依存関係'] != '-':
                dependencies = list(map(int, df.at[id, '依存関係'].split(',')))
                for dep in dependencies:
                    idx1 = get_s_index(variable_list, dep, n)
                    idx2 = get_s_index(variable_list, id, n)
                    if idx1 == -1 or idx2 == -1:
                        continue
                    A_cols.extend([idx1, idx2])
                    A_rows.extend([A_row, A_row])
                    A_row = A_row + 1
                    A_vals.extend([1, -1])
                    b_list.append(-E[dep])
                    label_list.append("③")

            # ④-1 同一タスクの一意割当制約 y(i, n, p1) + y(i, n, p2) … + y(i, n, p_max) <= NumP(i) 
            for p in Ps:
                index = get_y_index(variable_list, id, n, p)
                if index != -1:
                    A_cols.extend([get_y_index(variable_list, id, n, p)])
                    A_rows.extend([A_row])
                    A_vals.extend([1])
            A_row = A_row + 1
            b_list.append(NumP[id])
            label_list.append("④-1")

            # ④-2 同一タスクの一意割当制約 -{y(i, n, p1) + y(i, n, p2) … + y(i, n, p_max)} <= -NumP(i)
            for p in Ps:
                index = get_y_index(variable_list, id, n, p)
                if index != -1:
                    A_cols.extend([get_y_index(variable_list, id, n, p)])
                    A_rows.extend([A_row])
                    A_vals.extend([-1])
            A_row = A_row + 1
            b_list.append(-NumP[id])
            label_list.append("④-2")

            # 特定タスクのプロセッサに対しての割り当て禁止
            # そもそも変数を生成しないのでここではスキップ

    skip_list = []
    for x in variable_list['x']:
        # print(x)
        # ⑤-1 同一プロセッサ上の排他制約 s(j, m) - s(i, n) + M * x((i, n), (j, m))<= M - E(j)
        # ⑤-2 同一プロセッサ上の排他制約 s(i, n) - s(j, m) + M * x((j, m), (i, n))<= M - E(i)
        # (i,extend(j, m)の順番を入れ替えると同じ拘束条件になるため、ここでは片方のみ生成する
        index_dict = get_index_from_x(x)
        id1 = int(index_dict['ID1'])
        n1 = int(index_dict['n1'])
        id2 = int(index_dict['ID2'])
        n2 = int(index_dict['n2'])
        A_cols.extend([get_s_index(variable_list, id2, n2), get_s_index(variable_list, id1, n1), get_x_index_from_str(variable_list, x)])
        A_rows.extend([A_row, A_row, A_row])
        A_row = A_row + 1
        A_vals.extend([1, -1, M])
        b_list.append(M - E[id2])
        label_list.append("⑤-1")

        # ⑥-p 同一プロセッサ上のextend(i, n, p) + y(j, m, p) - 2*x((i, n),(j, m)) - 2*x((j, m),(i, n)) <= 1
        if not "{}:{}:{}:{}".format(id1, n1, id2, n2) in skip_list:
            for p in Ps:
                index1 = get_y_index(variable_list, id1, n1, p)
                index2 = get_y_index(variable_list, id2, n2, p)
                if index1 != -1 and index2 != -1:
                    A_cols.extend([get_y_index(variable_list, id1, n1, p),
                                   get_y_index(variable_list, id2, n2, p),
                                   get_x_index(variable_list, id1, n1, id2, n2),
                                   get_x_index(variable_list, id2, n2, id1, n1)])
                    A_rows.extend([A_row, A_row, A_row, A_row])
                    A_row = A_row + 1
                    A_vals.extend([1, 1, -2, -2])
                    b_list.append(1)
                    # (i, n)と(j, m)の順番を入れ替えたパターンはスキップリストに追加する
                    skip_list.append("{}:{}:{}:{}".format(id2, n2, id1, n1))
                    label_list.append("⑥-{}".format(p))

    A_coo = coo_matrix((A_vals, (A_rows, A_cols)), shape=(A_row, variable_num))
    A = A_coo.tocsc()
    b = np.array(b_list, dtype=float)

    # 標準形のAx ≤ bなのでlbはなし

    lb = [0.0] * variable_num
    ub = [float(M)] * variable_list['s'].__len__() + [float(M)] * variable_list['t'].__len__() + [1.0] * variable_list['x'].__len__() + [1.0] * variable_list['y'].__len__()
    integrality = np.array([0] * variable_list['s'].__len__() + [0] * variable_list['t'].__len__() + [3] * variable_list['x'].__len__() + [3] * variable_list['y'].__len__(), dtype=int)

    if verbose:
        with open('target.txt', 'w') as f:
            print('A_rows {}'.format(len(A_rows)), file=f)
            print(A_rows, file=f)
            print('A_rows_boundaries', file=f)
            print(boundaries(A_rows), file=f)
            print('A_cols {}'.format(len(A_cols)), file=f)
            print(A_cols, file=f)
            print('A_val {}'.format(len(A_vals)), file=f)
            print(list(map(float, A_vals)), file=f)
            print('b {}'.format(len(b)), file=f)
            print(list(map(float, b)), file=f)
            print('c {}'.format(len(c)), file=f)
            print(list(map(float, c)), file=f)
            print('integrality {} ~ {}'.format(variable_list['s'].__len__() + variable_list['t'].__len__(), variable_list['s'].__len__() + variable_list['t'].__len__() + variable_list['x'].__len__() + variable_list['y'].__len__()), file=f)
            print(A, file=f)

    # mpsファイルで問題を出力
    generate_mps_file(variable_num, variable_list, A, b, c)

    ret = True
    if variable_num > 10000:
        ret = confirm("大規模なMILPを解くためPCの動作に影響する可能性があります。実行しますか？ [y/N]: ", default=False)

    if ret:
        h = Highs()
        h.readModel('./taskschedule.mps')
        h.run()
        ret = h.getModelStatus()

        if ret in [HighsModelStatus.kOptimal]:
            sol = h.getSolution()
            sol_var = list(sol.col_value)
            sol_fun = h.getInfo().objective_function_value
            df_res = pd.DataFrame (data=[sol_var], columns=variable_list['s'] + variable_list['t']+ variable_list['x'] + variable_list['y'])
            print('result :', ret)
            print(sol_var)
            print(sol_fun)
            return True, df_res, variable_list
        else:
            print('result:', ret)
            return False, None, None
    else:
        print('中止しました。')
        return False, None, None


# ガントチャート描画
def plot_gantt_from_df(df_tasks, df_result, variable_list,processor_num=0):
    # 定数設定
    # タスクID
    IDs = df_tasks['ID'].values.tolist()
    # 実行時間
    E = df_tasks['実行時間E(i)'].values.tolist()
    # 占有プロセッサ数
    if processor_num == 0:
        processor_num = len(variable_list['P'])
    print('processor_num:{}'.format(processor_num))
    # プロセッサ数
    Ps = [i for i in range(processor_num)]

    variable_num = len(variable_list['s']) + len(variable_list['t']) + len(variable_list['y']) + len(variable_list['x'])
    print('variable_num: {}, Hyperperiod: {}'.format(variable_num, variable_list['M']))

    tasks = []
    start = 0
    end = 0
    for id in IDs:
        for n in variable_list['n_range'][id]:
            for p in Ps:
                col = 'y(({}, {}), {})'.format(id, n, p)
                if col in df_result.columns:
                    if df_result[col].values == 1:
                        # プロセスpに対してタスクidのn回目が割り当てられている場合
                        start = float(df_result.at[0, 's({}, {})'.format(id, n)])
                        end = start + E[id]
                        tasks.append({'task': id, 'p': p, 'n': n, 'start':start, 'end':end, 'label': 'task_{}_{}'.format(id, n), 'dead':n*df_tasks.at[id, 'デッドラインD(i)']})

    with open('gantt_tasks.txt', 'w') as f:
        print(tasks,file=f)

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=(16, 4))


    # --- 色の割り当て（タスクIDごとに固定色） ---
    unique_ids = sorted(set(t['task'] for t in tasks))
    cmap = plt.get_cmap('tab10')  # 見やすい10色
    task_to_color = {tid: cmap(i % 20) for i, tid in enumerate(unique_ids)}

    for t in tasks:
        width = t["end"] - t["start"]
        ax.barh(y=t["p"], width=width, left=t["start"], height=0.6, align='center', color=task_to_color[t["task"]])
        ax.text(t["start"] + width/2, t["p"], t["label"], ha="center", va="center", color="white")
        ax.axvline(x=t["dead"], color=task_to_color[t["task"]], linestyle='--')

    # 軸設定
    ax.set_xlabel("time")
    ax.set_ylabel("Processor")
    ax.set_yticks(Ps)
    ax.set_yticklabels([f"P{p}" for p in Ps])
    ax.grid(axis="x", linestyle=":", color="gray")

    plt.title("Sample")
    plt.savefig("figure.png", dpi=200, bbox_inches="tight")  # ← show() の代わり
    plt.show()

"""
ランダムにタスクセットCSVを生成する（コマンドライン引数なし）。
出力フォーマット:
ID,周期T(i),実行時間E(i),デッドラインD(i),依存関係,専用プロセッサリスト,使用プロセッサ数P(i)
"""

import csv
import random
from typing import List, Optional


def rand_period(min_t: int = 5, max_t: int = 100, step:int = 1) -> int:
    """周期 T(i) をランダムに生成。"""
    rand_min = max(1, min_t)
    return random.randint(rand_min, max_t ) * step


def rand_exec_time(T: int) -> int:
    """実行時間 E(i) を T 以下でランダム生成（最低1）。"""
    return random.randint(1, max(1, int(T * 0.35)))  # 過度に長くならないように半分程度を上限に


def rand_deadline(E: int, T: int) -> int:
    """デッドライン D(i) を E..T の範囲で生成。"""
    return random.randint(E, T)


def choose_dependencies(cur_id: int, dep_prob: float = 0.4) -> Optional[List[int]]:
    """
    依存関係: 現在IDより小さいIDからランダム選択。
    dep_prob: 依存関係を持つ確率。
    """
    if cur_id == 0 or random.random() > dep_prob:
        return None  # "-" を意味する
    candidates = list(range(0, cur_id))
    k = random.randint(1, min(len(candidates), 2))  # 多すぎないよう最大3
    return sorted(random.sample(candidates, k))


def choose_dedicated_processors(max_procs: int, use_prob: float = 0.5) -> Optional[List[int]]:
    """
    専用プロセッサリスト: 0..(max_procs-1) からランダム選択。
    use_prob: リストを作る（"-"でない）確率。
    """
    if random.random() > use_prob or max_procs <= 0:
        return None  # "-" を意味する
    k = random.randint(1, min(max_procs, 4))
    return sorted(random.sample(range(max_procs), k))


def format_list_or_dash(items: Optional[List[int]]) -> str:
    """CSV用の文字列に整形。None は "-"、リストはカンマ区切り文字列に。"""
    if not items:
        return '-'
    return str(items).strip('[]').replace(' ', '')


def generate_tasks(num_tasks: int, max_procs: int,
                   min_t: int, max_t: int,cycle_step: int,
                   dep_prob: float, use_prob: float) -> List[List[str]]:
    """タスクリストを生成して、CSV行（文字列）として返す。"""
    rows: List[List[str]] = []
    for i in range(num_tasks):
        T = rand_period(min_t=min_t, max_t=max_t, step=cycle_step)
        E = rand_exec_time(T)
        #D = rand_deadline(E, T)
        D = T

        deps = choose_dependencies(cur_id=i, dep_prob=dep_prob)
        deps_str = format_list_or_dash(deps)

        dedicated = choose_dedicated_processors(max_procs=max_procs, use_prob=use_prob)
        dedicated_str = format_list_or_dash(dedicated)

        # 使用プロセッサ数 P(i): 専用リストがある場合はその数、ない場合は 1..max_procs のランダム
        if dedicated:
            P = len(dedicated)
        else:
            #P = random.randint(1, max(1, max_procs))
            P = random.choices([1, 2, 3, 4], weights=[0.5, 0.25, 0.125, 0.0625], k=1)[0]

        row = [
            str(i),               # ID
            str(T),               # 周期T(i)
            str(E),               # 実行時間E(i)
            str(D),               # デッドラインD(i)
            deps_str,             # 依存関係（カンマ区切り or "-" をダブルクォートで囲む）
            dedicated_str,        # 専用プロセッサリスト
            str(P),               # 使用プロセッサ数P(i)
        ]
        rows.append(row)
    return rows


def generate_tasks_main():
    # ===== ここでパラメータを設定（必要に応じて変更してください） =====
    NUM_TASKS = 6           # 生成するタスク数
    MAX_PROCS = 4           # プロセッサ総数（0..MAX_PROCS-1）
    OUTFILE   = "tasks.csv" # 出力CSVファイル名
    SEED      = None        # 乱数シード（Noneならランダム）

    # 生成パラメータ（分布の調整用）
    MIN_T     = 1           # 周期T(i)の最小
    MAX_T     = 3           # 周期T(i)の最大
    CYCLE_STEP = 16         # 周期T(i)の刻み幅
    DEP_PROB  = 0.2         # 依存関係を持つ確率 [0..1]
    DEDICATED_PROB = 0.2    # 専用プロセッサリストを持つ確率 [0..1]
    # =========================================================

    if SEED is not None:
        random.seed(SEED)

    rows = generate_tasks(
        num_tasks=NUM_TASKS,
        max_procs=MAX_PROCS,
        min_t=MIN_T,
        max_t=MAX_T,
        cycle_step=CYCLE_STEP,
        dep_prob=DEP_PROB,
        use_prob=DEDICATED_PROB,
    )

    header = ["ID", "周期T(i)", "実行時間E(i)", "デッドラインD(i)", "依存関係", "専用プロセッサリスト", "使用プロセッサ数P(i)"]
    with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {OUTFILE} with {len(rows)} tasks.")


if __name__ == "__main__":
    processor_num = 5

    if len(sys.argv) == 1:
        generate_tasks_main()
        # check
        df_task = pd.read_csv('tasks.csv', header=0)
    elif len(sys.argv) <= 3:
        df_task = pd.read_csv(sys.argv[1], header=0)
        if len(sys.argv) == 3:
            if sys.argv[2].isdigit():
                processor_num = int(sys.argv[2])
    else:
        print('Usage: python MILP.py [task_file.csv]')
        exit(1)

    result, df_result, variable_list = solve_MILP_csc(df_task, processor_num=processor_num)
    if result:
        df_result.to_csv('result_tasks.csv', index=False)
        plot_gantt_from_df(df_task, df_result, variable_list)
    else:
        print('解が見つかりませんでした。')

