"""Render the Chinese frozen-prediction diagnosis from the publishable tables."""

import csv
import json
import math
from pathlib import Path


FOLDS = ('hotel', 'univ')
TEST = 'heldout_confirmation'
DEV = 'selection_dev'
SPLIT_LABELS = {TEST: '测试集', DEV: '开发集', 'train': '训练集'}
GROUP_LABELS = {
    'all': '全体', 'N1': '单人', 'N2_8': '2–8 人', 'N9plus': 'N≥9',
    'top10': '头部 10%', 'remaining90': '其余窗',
    'top10_N1': '头部单人', 'top10_N2_8': '头部 2–8 人', 'top10_N9plus': '头部 N≥9',
}
MAIN_GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
TERM = ('J', 'L', 'G', 'minFDE')
GFTD = {'hotel': (.174, .297), 'univ': (.649, 1.305)}
GFTD_PAPER = 'https://arxiv.org/html/2404.00237v2'


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def _number(row, key):
    value = row.get(key)
    if value is None or value == '':
        return math.nan
    return float(value)


def _fmt(value, digits=4, signed=False):
    if not math.isfinite(value):
        return 'N/A'
    if value == 0:
        value = 0.
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def _fields(row, keys, *, digits=4, scale=1., signed=False):
    return ' / '.join(_fmt(_number(row, key)*scale, digits, signed) for key in keys)


def _count(row, field='windows'):
    value = _number(row, field)
    return f'{int(value):,}' if math.isfinite(value) else 'N/A'


def _size(row):
    return _count(row) + '／' + _count(row, 'agents')


def _table(headers, rows):
    def line(cells):
        return '| ' + ' | '.join(cell.replace('|', '\\|') for cell in cells) + ' |'

    return '\n'.join([
        line(headers),
        '|' + '|'.join('---' for _ in headers) + '|',
        *[line(row) for row in rows],
    ])


def _share(row, total):
    if not _number(row, 'windows'):
        return 0.
    return 100 * (_number(row, 'windows') * _number(row, 'g0_J_m')
                  / (_number(total, 'windows') * _number(total, 'g0_J_m')))


def _headers(groups):
    hotel = groups[TEST, 'hotel', 'all']
    singleton = groups[TEST, 'hotel', 'N1']
    tail = groups[TEST, 'hotel', 'top10']
    multi = groups[TEST, 'hotel', 'N2_8']
    univ = groups[TEST, 'univ', 'all']
    dense = groups[TEST, 'univ', 'N9plus']
    tail_l = _number(tail, 'g0_L_m')
    tail_g = _number(tail, 'g0_G_m')
    univ_dev_dense = groups[DEV, 'univ', 'N9plus']
    univ_dev = groups[DEV, 'univ', 'all']
    gap_share = 100 * _number(dense, 'g0_G_m') / _number(dense, 'g0_J_m')
    near = _number(dense, 'g0_JP_m')
    random = _number(dense, 'g0_random_JP_m')
    within = _number(dense, 'g0_JP_minus_L_m')
    return [
        '## HOTEL',
        f'1. **误差集中在哪些窗**　{_count(multi)} 个多人测试窗贡献全折 '
        f'{_share(multi, hotel):.1f}% 的 S0 JFDE 总和。S0 误差最高的 {_count(tail)} 窗贡献 '
        f'{_share(tail, hotel):.1f}%。',
        f'2. **最早在哪层看到**　这批高误差窗的 raw J / L / G 已达 '
        f'{_fields(tail, ("raw_J_m", "raw_L_m", "raw_G_m"), digits=3)} m。候选覆盖不足和槽位共用代价均已出现。',
        f'3. **量级**　旧头后的高误差窗 J 为 {tail_l + tail_g:.3f} m，其中 L 为 {tail_l:.3f} m、'
        f'G 为 {tail_g:.3f} m。旧头和 R12 分别改变 J '
        f'{_fields(tail, ("delta_g0_raw_J_cm", "delta_g1_g0_J_cm"), signed=True)} cm。',
        f'4. **限制这项解释的证据**　高误差窗的 G 为 {tail_g/(tail_l+tail_g):.1%}，'
        f'多人尾部的误差需要同时考虑槽位。{_count(singleton)} 个单人窗的 L 仍为 '
        f'{_number(singleton, "g0_L_m"):.3f} m，给出了单人覆盖误差。',
        '## UNIV',
        f'1. **误差集中在哪些窗**　{_count(dense)} 个 N≥9 测试窗贡献全折 '
        f'{_share(dense, univ):.1f}% 的 S0 JFDE 总和。该占比同时受密集窗数量和窗内误差影响。',
        f'2. **最早在哪层看到**　这些密集窗的 raw J / L / G 为 '
        f'{_fields(dense, ("raw_J_m", "raw_L_m", "raw_G_m"), digits=3)} m。共用槽位的代价在原生 K20 即可测得。',
        f'3. **量级**　旧头后的密集窗 J / L / G 为 '
        f'{_fields(dense, ("g0_J_m", "g0_L_m", "g0_G_m"), digits=3)} m。'
        f'G 占 J 的 {gap_share:.1f}%，个人覆盖误差仍有 {_number(dense, "g0_L_m"):.3f} m。',
        f'4. **限制这项解释的证据**　开发集仅有 {_count(univ_dev_dense)}／{_count(univ_dev)}'
        f' 个密集窗。测试密集窗近邻 Jₚ 为 {_fmt(near)} m，同组大小随机参照为 {_fmt(random)} m。'
        f'近邻组内共用槽位仍付出 {100*within:.2f} cm，逐人放松后的 L 仍有 '
        f'{_number(dense, "g0_L_m"):.3f} m。',
    ]


def _endpoints(groups):
    rows = []
    for split in (TEST, DEV):
        for fold in FOLDS:
            for group in MAIN_GROUPS:
                row = groups[split, fold, group]
                rows.append([
                    SPLIT_LABELS[split], fold.upper(), GROUP_LABELS[group], _size(row),
                    *[_fields(row, [f'{stage}_{term}_m' for term in TERM])
                      for stage in ('raw', 'g0', 'g1')],
                    *[_fields(row, [f'delta_{stage}_{term}_cm' for term in TERM], signed=True)
                      for stage in ('g0_raw', 'g1_g0')],
                ])
    return [
        '## 端点误差从哪一层开始',
        'J 是所有行人共用最优槽位的终点误差，L 是每人各取最优终点后的窗均值，G=J−L。'
        '前三项按窗等权，minFDE 按有效人次加权。端点对齐各归档种子后，每窗只计一次。'
        'raw 为 Flow 输出，g0 为 S0 原有候选头输出，g1 为 R12 修正后终点。',
        _table(['划分', '折', '人数', '窗／人次', 'raw（m）', 'g0（m）', 'g1（m）',
                'g0−raw（cm）', 'g1−g0（cm）'], rows),
        '各指标单元格顺序为 J / L / G / minFDE。末帧锚定的核验见本页末尾，'
        '终点和联合槽位的变化可直接追溯 JFDE。',
    ]


def _r12(groups, windows):
    rows = []
    for split in (TEST, DEV):
        for fold in FOLDS:
            for group in ('all', 'N2_8', 'N9plus'):
                row = groups[split, fold, group]
                if not _number(row, 'windows'):
                    continue
                rows.append([
                    SPLIT_LABELS[split], fold.upper(), GROUP_LABELS[group],
                    _fields(row, ['H_median_m', 'H_p90_m', 'H_max_m'], scale=100),
                    *[_fields(row, [f'{branch}_median_m', f'{branch}_p90_m'], scale=100)
                      for branch in ('unary', 'social', 'combined')],
                    _fields(row, ['abs_delta_J_max_m', 'winner_bound_max_m'], scale=100),
                    _count(row, 'H_ge_1cm_windows') + '／' + _count(row, 'abs_delta_J_ge_1cm_windows'),
                    _count(row, 'H_bound_violation_windows'),
                ])
    pieces = []
    for split in (TEST, DEV):
        for fold in FOLDS:
            for group in ('all', 'N2_8', 'N9plus'):
                row = groups[split, fold, group]
                if not _number(row, 'windows'):
                    continue
                pieces.append([
                    SPLIT_LABELS[split], fold.upper(), GROUP_LABELS[group],
                    _fields(row, ['winner_delta_fixed_cm', 'winner_delta_switch_cm'], signed=True),
                    _fields(row, ['winner_switch_rate', 'cancel_ratio', 'branch_opposition_rate'],
                            scale=100, digits=2),
                    _fields(row, ['winner_shift_m', 'nonwinner_shift_m'], scale=100),
                    _fields(row, ['winner_toward_projection_mean_m'], scale=100, signed=True),
                    _fields(row, ['winner_toward_positive_rate'], scale=100, digits=2),
                ])
    notes = []
    for fold in FOLDS:
        row = groups[TEST, fold, 'all']
        high = _number(row, 'H_ge_1cm_windows')
        limit = _number(row, 'H_max_m') * 100
        maximum = _number(row, 'abs_delta_J_max_m') * 100
        winner_bound = _number(row, 'winner_bound_max_m') * 100
        win_move = _number(row, 'winner_shift_m') * 100
        other_move = _number(row, 'nonwinner_shift_m') * 100
        projection = _number(row, 'winner_toward_projection_mean_m') * 100
        exceptions = [win for win in windows if win['split'] == TEST and win['fold'] == fold
                      and _number(win, 'H_m') >= .01]
        if len(exceptions) != int(high):
            raise ValueError('The per-window and group H≥1 cm counts differ')
        detail = ''
        if exceptions:
            exceptional_move = 100 * max(_number(win, 'winner_shift_m') for win in exceptions)
            exceptional_j = 100 * max(_number(win, 'abs_delta_J_m') for win in exceptions)
            switches = sum(_number(win, 'winner_switch_rate') for win in exceptions)
            counts = {group: sum(win['group'] == group for win in exceptions)
                      for group in MAIN_GROUPS[1:]}
            composition = '、'.join(
                f'{GROUP_LABELS[group]} {count} 窗' for group, count in counts.items() if count)
            detail = (f' 其中{composition}，旧赢家最大人均位移 {exceptional_move:.4f} cm，'
                      f'换赢家 {int(switches)} 窗，实测最大 |ΔJ| {exceptional_j:.4f} cm。')
        switch = abs(_number(row, 'winner_delta_switch_cm'))
        fixed = _number(row, 'winner_delta_fixed_cm')
        if abs(fixed) >= switch:
            attribution = f'平均固定赢家变化 {fixed:+.4f} cm，换赢家额外降低 {switch:.4f} cm。'
        else:
            attribution = f'换赢家额外降低 {switch:.4f} cm，大于固定赢家项的绝对幅度 {abs(fixed):.4f} cm。'
        notes.append(
            f'- **{fold.upper()}**　实测最大 |ΔJ| 为 {maximum:.4f} cm。'
            + (f'两赢家上界最大 {winner_bound:.4f} cm。' if math.isfinite(winner_bound) else '')
            + (f'H 最大仅 {limit:.4f} cm，逐窗位移上界已解释 1 cm 以内的变化。'
               if high == 0 else
               f'{int(high)} 窗的 H 达到 1 cm，单用 H 仍容许这些窗发生更大变化。')
            + f' 原赢家人均位移 {win_move:.4f} cm，其他槽 {other_move:.4f} cm，'
            f'原赢家朝真值的平均带符号投影 {projection:+.4f} cm。'
            + attribution + detail
        )
    return [
        '## R12 的影响为什么小',
        '对每个窗口和每个槽，先对人的位移长度求平均，再取跨槽最大值 H。'
        '由三角不等式，|ΔJ|≤H。H 的分位数按窗计算，各分支位移长度的分位数按槽×人计算。',
        '令 D[k] 为第 k 槽的人均位移。用旧赢家 k₀ 控制新目标的最小值，有 J₁−J₀≤D[k₀]。'
        '用新赢家 k₁ 控制旧目标的最小值，有 J₀−J₁≤D[k₁]。'
        '因此 max(D[k₀], D[k₁]) 是更紧的两赢家上界。',
        _table(['划分', '折', '人数', 'H 中位 / P90 / 最大（cm）', '个人 中位 / P90（cm）',
                '交互 中位 / P90（cm）', '合成 中位 / P90（cm）',
                '|ΔJ| / 两赢家上界 最大（cm）', 'H≥1 cm／|ΔJ|≥1 cm 窗数', '越 H 窗数'], rows),
        '用修正前的赢家 k₀ 和修正后的赢家 k₁，将 ΔJ 精确拆成固定赢家的误差变化'
        ' C₁[k₀]−C₀[k₀]，加上换赢家的额外收益 C₁[k₁]−C₁[k₀]。'
        '正投影表示向真值移动。抵消率为 1−Σ‖合成‖/Σ(‖个人‖+‖交互‖)。',
        _table(['划分', '折', '人数', '固定赢家 / 换赢家（cm）', '换赢家 / 抵消 / 反向比例（%）',
                '赢家 / 其他槽位移（cm）', '赢家朝真值投影（cm）', '正投影（%）'], pieces),
        '\n'.join(notes),
        '这些量反映实际修正的位移尺度、方向与槽位选择。学习率或保护项的作用需要独立干预才能识别。',
    ]


def _hotel(groups):
    rows = []
    for split in (TEST, DEV):
        total = groups[split, 'hotel', 'all']
        for group in ('N1', 'N2_8', 'N9plus', 'top10', 'remaining90', 'top10_N1',
                      'top10_N2_8', 'top10_N9plus'):
            row = groups.get((split, 'hotel', group), {})
            if not _number(row, 'windows') > 0:
                continue
            rows.append([
                SPLIT_LABELS[split], GROUP_LABELS[group], _size(row), f'{_share(row, total):.1f}',
                _fields(row, ('g0_J_m', 'g0_L_m', 'g0_G_m', 'cv_FDE_m')),
                _count(row, 'axis_agents') + '／' + _count(row, 'low_speed_agents'),
                _fields(row, ('g0_joint_long_signed_m', 'g0_joint_lat_signed_m'), signed=True),
                _fields(row, ('g0_joint_long_abs_m', 'g0_joint_lat_abs_m')),
                _fields(row, ('g0_person_long_abs_m', 'g0_person_lat_abs_m')),
                _fields(row, ('cv_long_abs_m', 'cv_lat_abs_m')),
                _fields(row, ('g0_joint_no_axis_FDE_m', 'g0_person_no_axis_FDE_m', 'cv_no_axis_FDE_m')),
            ])
    motion = []
    for split in (TEST, DEV):
        for group in ('all', 'N1', 'N2_8', 'top10'):
            row = groups[split, 'hotel', group]
            for key, label in (
                ('stay_slow', '持续低速'), ('start', '由低速转运动'), ('stop', '由运动转低速'),
                ('keep_moving', '持续运动'),
            ):
                prefix = f'motion_{key}_'
                if not _number(row, prefix+'agents') > 0:
                    continue
                motion.append([SPLIT_LABELS[split], GROUP_LABELS[group], label, _count(row, prefix+'agents'),
                    _fields(row, [prefix+'g0_joint_FDE_m', prefix+'g0_person_FDE_m', prefix+'cv_FDE_m'])])
    lines = [
        '## HOTEL 高误差窗与速度、朝向',
        'S0 的 JFDE 从高到低取最高 10%，向上取整，同分按匿名窗号。'
        '窗级贡献为组内 S0 JFDE 之和除以全折同项之和。',
        _table(['划分', '组', '窗／人次', '贡献（%）', 'J / L / G / CV（m）',
                '可投影／低速人次', 'S0 联合 纵／横带符号（m）', 'S0 联合 纵／横绝对值（m）',
                'S0 逐人 纵／横绝对值（m）', 'CV 纵／横绝对值（m）',
                '无朝向 联合／逐人／CV（m）'], rows),
        'CV 将最后三段观测的平均速度外推 12 帧。CV 误差按窗等权。方向相对观测朝向，'
        '终点预测比真值更靠前时纵向为正。观测路程均速不超过 0.2 m/s'
        ' 或末三段矢量均速不超过 0.001 m/s 的行人只计欧氏误差。',
    ]
    tail = groups[TEST, 'hotel', 'top10']
    speed = _fields(tail, ('observed_speed_mean_mps', 'future_speed_mean_mps'), digits=3)
    axes = _fields(tail, ('g0_person_long_abs_m', 'g0_person_lat_abs_m'), digits=3)
    signed = _number(tail, 'g0_joint_long_signed_m')
    direction = '向前偏' if signed > 0 else '向后偏' if signed < 0 else '接近零'
    lines.append(
        f'测试头部 10% 中，观测／未来平均路程速度为 {speed} m/s，'
        f'未来位移相对观测朝向的平均绝对转角为 {_fmt(_number(tail, "truth_turn_abs_deg"), 1)}°。'
        f'S0 联合赢家的平均纵向误差为 {_fmt(signed, 3, True)} m，{direction}。'
        f'S0 逐人择槽的纵／横向绝对误差仍有 {axes} m。'
        f'在这批窗里，CV 的平均终点误差低于 S0 联合 J、逐人 L 的比例分别为 '
        f'{_fields(tail, ("cv_beats_g0_J_rate", "cv_beats_g0_L_rate"), digits=1, scale=100)}%。'
    )
    if motion:
        lines.extend([
            '下表分别按观测与真值时段的总路程除以时长，以 0.2 m/s 为固定低速界，标记停走变化。'
            '每行比较同一批行人的欧氏误差。',
            _table(['划分', '组', '运动变化', '人次', 'S0 联合 / S0 逐人 / CV FDE（m）'], motion),
        ])
    return lines


def _partitions(groups, demography):
    population = [row for row in demography if row['fold'] == 'univ']
    rows = []
    for split in ('train', DEV, TEST):
        subrows = {row['group']: row for row in population if row['split'] == split}
        total = subrows['all']
        dense = subrows['N9plus']
        rows.append([
            SPLIT_LABELS[split], _size(total),
            *[_size(subrows[group]) for group in MAIN_GROUPS[1:]],
            f'{_number(dense, "windows")/_number(total, "windows"):.1%}',
        ])
    partition = []
    for split in (TEST, DEV):
        for fold in FOLDS:
            for group in ('N2_8', 'N9plus'):
                row = groups[split, fold, group]
                if not _number(row, 'windows'):
                    continue
                partition.append([
                    SPLIT_LABELS[split], fold.upper(), GROUP_LABELS[group], _count(row),
                    _count(row, 'partition_informative_windows'),
                    _fields(row, ('pair_coverage',), scale=100, digits=2),
                    _fields(row, ('g0_JP_m', 'g0_J_minus_JP_m', 'g0_JP_minus_L_m')),
                    _fields(row, ('g0_random_JP_m', 'g0_random_J_minus_JP_m',
                                  'g0_random_JP_minus_L_m')),
                ])
    return [
        '## UNIV 人数分布与两折局部槽位松弛',
        _table(['UNIV 划分', '窗／人次', '单人', '2–8 人', 'N≥9', '密集窗比例'], rows),
        '对两折多人窗，仅用最后三帧观测的平均两两距离，在 2 m 内从小到大贪心配对，'
        '每人最多一对。未入对的行人单列。相同组大小的随机配对固定做 20 次，先在窗内求平均。'
        '没有中间粗细层级的划分标为无区分力。表中代价汇总使用有区分力的窗，覆盖率按分组表口径。',
        _table(['划分', '折', '组', '窗', '有区分力窗', '配对覆盖（%）',
                '近邻 Jₚ / J−Jₚ / Jₚ−L（m）', '随机 Jₚ / J−Jₚ / Jₚ−L（m）'], partition),
        '各组分别用未来真值择优得到 Jₚ，仅表示固定终点集上放松槽位约束的诊断空间。'
        'J−Jₚ 表示跨组共用槽位的代价，Jₚ−L 表示组内共用槽位的代价。',
    ]


def _paths(groups):
    rows = []
    for split in (TEST, DEV):
        for fold in FOLDS:
            group = groups[split, fold, 'all']
            for model in ('S0', 'R12'):
                for winner in ('JADE', 'JFDE'):
                    keys = [f'path_{model}_{winner}_winner_t{frame}' for frame in (3, 6, 9, 12)]
                    rows.append([
                        SPLIT_LABELS[split], fold.upper(), model, winner,
                        _fields(group, [key+'_error_m' for key in keys]),
                        _fields(group, [key+'_q_m' for key in keys]),
                    ])
    residual = []
    for fold in FOLDS:
        row = groups[TEST, fold, 'all']
        values = _fields(row, [f'path_S0_JADE_winner_t{frame}_q_m' for frame in (3, 6, 9)], digits=3)
        residual.append(f'{fold.upper()} 为 {values} m')
    return [
        '## 固定赢家后的中途偏差',
        '分别固定最优 JADE 和最优 JFDE 的整场槽，对每窗先汇总各推理种子。'
        'q 是同槽误差向量扣除按帧比例线性分摊的末帧误差向量，'
        '其范数量化剩余形状偏差。因果归因到轨迹桥需要固定端点的干预。',
        _table(['划分', '折', '模型', '固定赢家', '第 3 / 6 / 9 / 12 帧误差（m）',
                '第 3 / 6 / 9 / 12 帧 ‖q‖（m）'], rows),
        'S0 的 JADE 赢家即使扣除线性末帧误差，测试集第 3／6／9 帧仍有形状偏差。'
        + '，'.join(residual) + '。',
    ]


def _published_reference(groups):
    rows = []
    for fold in FOLDS:
        row = groups[TEST, fold, 'all']
        paper = GFTD[fold]
        ours = [_number(row, f'path_S0_{metric}_m') for metric in ('JADE', 'JFDE')]
        rows.append([fold.upper(), _fields(row, ('path_S0_JADE_m','path_S0_JFDE_m')),
            _fields(row, ('path_R12_JADE_m','path_R12_JFDE_m')),
            ' / '.join(_fmt(x, 3) for x in paper),
            ' / '.join(_fmt(100*(a-b), 2, True) for a, b in zip(ours, paper))])
    return [
        '## 论文表 I 的同折参照',
        _table(['折', 'S0 JADE / JFDE（m）', 'R12 JADE / JFDE（m）',
                'GFTD 表 I JADE / JFDE（m）', 'S0−表 I（cm）'], rows),
        '两边均为 ETH/UCY 同名折、8 帧观测、12 帧预测与 K20。'
        '本文按仓库原生 JFDE 归约，论文式 (17) 的末帧项前印为 1/(TN)。'
        '外部数字按表 I 原值列示，缺少与本次封存窗口配对的 GFTD 逐窗轨迹。'
        f' [Lin 等，论文表 I 与 §V-A]({GFTD_PAPER})',
    ]


def _cases(rows):
    links = []
    for row in rows:
        kind = row.get('case') or row.get('kind') or row.get('criterion', '')
        kind = {'highest_L': '高 L', 'highest_G': '高 G', 'median_J': '中位 J'}.get(kind, kind)
        name = ' '.join(filter(None, (row['fold'].upper(), kind, row.get('window', ''))))
        links.append(f'[{name}]({row["figure"]})')
    return [
        '## 匿名案例',
        '每折选 seed 2035 的高 L、高 G 和最接近 S0 JFDE 中位数的窗口。'
        '各图显示历史、真值、S0 的 JFDE 整场赢家与逐人最近终点对应轨迹。',
        ' · '.join(links),
    ]


def _integrity(receipt):
    def maximum(value):
        if isinstance(value, dict):
            return max(map(maximum, value.values()), default=0.)
        if value is None:
            return 0.
        return float(value)

    entries = [
        '## 运行核验',
        '固定入口仅导出原生 raw、g0、g1 与分支位移，预测封存后由评分入口读取真值。'
        '测试轨迹使用五个冻结种子，同窗先平均；开发集复用种子 2035。零训练。',
        f'归档指标最大绝对差 {maximum(receipt["replay_max_abs_m"]):.4g} m，'
        f'末帧锚定最大绝对差 {maximum(receipt["endpoint_replay_max_abs_m"]):.4g} m，'
        f'端点跨种子最大绝对差 {maximum(receipt["cross_seed_endpoint_max_abs_m"]):.4g} m，'
        f'位移分支分解最大绝对差 {maximum(receipt["decomposition_max_abs_m"]):.4g} m。'
        f'新增 DDIM 调用 {int(receipt["new_ddim_calls"])} 次。',
    ]
    timings = receipt.get('seconds')
    if isinstance(timings, dict):
        entries.append('运行耗时（秒） ' + '，'.join(
            f'{name} {float(seconds):.1f}' for name, seconds in timings.items()
            if isinstance(seconds, (int, float))) + '。')
    elif isinstance(timings, (int, float)):
        entries.append(f'耗时 {timings:.1f} 秒。')
    entries.append('[分组汇总](GROUPS.csv) · [匿名逐窗](PER_WINDOW.csv) · '
                   '[人数分布](DEMOGRAPHY.csv) · [案例索引](CASES.csv) · [运行收据](RECEIPT.json)')
    return entries


def _next_experiment(groups):
    dense = groups[TEST, 'univ', 'N9plus']
    hotel_tail = groups[TEST, 'hotel', 'top10']
    scope = ('UNIV 密集窗的原生 G 较大，且固定端点后已有可测的槽位松弛空间。'
             'HOTEL 高误差窗同时有覆盖不足，用两折独立结果衡量迁移效果。')
    if _number(dense, 'g0_G_m') < _number(dense, 'g0_L_m') and (
            _number(hotel_tail, 'g0_G_m') < _number(hotel_tail, 'g0_L_m')):
        return [
            '## 唯一后续验证',
            '**优先在开发集检验原生候选覆盖。** 固定槽位和桥，'
            '只将每人的一条候选终点换为观测匀速外推，按原协议比较 L、G 与 J。'
            '这样可测单纯改变候选内容能否同时降低 L 和 J。',
        ]
    return [
        '## 唯一后续验证',
        '**优先验证联合槽位构造。** ' + scope +
        '在原开发集保持每人 K20 终点集合固定，只用观测和候选终点重新构造共享槽位，'
        '以逐人 minFDE 恒等和联合 JFDE 的配对变化验收。先做端点评分即可检验这项改动。',
    ]


def _development(groups):
    hotel_test = groups[TEST, 'hotel', 'top10']
    hotel_dev = groups[DEV, 'hotel', 'top10']
    univ_test = groups[TEST, 'univ', 'N9plus']
    univ_dev = groups[DEV, 'univ', 'N9plus']
    return [
        '## 开发集复核',
        f'HOTEL 开发集高误差窗在 raw 即有较大的 L / G，分别为 '
        f'{_fields(hotel_dev, ("raw_L_m", "raw_G_m"), digits=3)} m。'
        f'其中 R12−S0 的平均 JFDE 差为 {_fmt(_number(hotel_dev, "delta_g1_g0_J_cm"), signed=True)} cm，'
        f'对应测试头部为 {_fmt(_number(hotel_test, "delta_g1_g0_J_cm"), signed=True)} cm。',
        f'UNIV 开发集密集窗的 g0 L / G 为 {_fields(univ_dev, ("g0_L_m", "g0_G_m"), digits=3)} m，'
        f'测试密集窗为 {_fields(univ_test, ("g0_L_m", "g0_G_m"), digits=3)} m。'
        f'开发集的 G−L 为 {_number(univ_dev, "g0_G_m")-_number(univ_dev, "g0_L_m"):.3f} m，'
        f'测试为 {_number(univ_test, "g0_G_m")-_number(univ_test, "g0_L_m"):.3f} m。'
        '原生槽位误差在两种划分中均可追溯，密集窗误差构成的差距提示开发集验证需保留人数分组。',
    ]


def build_report(doc: Path):
    """Render from the scored and scrubbed public artifacts."""
    doc = Path(doc)
    receipt = json.loads((doc / 'RECEIPT.json').read_text(encoding='utf-8'))
    groups = {(r['split'], r['fold'], r['group']): r for r in _read_csv(doc / 'GROUPS.csv')}
    windows = _read_csv(doc / 'PER_WINDOW.csv')
    demography = _read_csv(doc / 'DEMOGRAPHY.csv')
    cases = _read_csv(doc / 'CASES.csv')
    paragraphs = [
        '# HOTEL 与 UNIV 冻结预测误差诊断',
        'HOTEL 的 S0 为 S24，UNIV 的 S0 为 S16。两折分别与已封存 R12 配对。'
        '首页使用测试集诊断，开发集复核与五种子路径汇总列在后文。',
        *_headers(groups),
        *_endpoints(groups),
        *_r12(groups, windows),
        *_hotel(groups),
        *_partitions(groups, demography),
        *_paths(groups),
        *_development(groups),
        *_published_reference(groups),
        *_cases(cases),
        *_integrity(receipt),
        *_next_experiment(groups),
    ]
    (doc / 'README.md').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
