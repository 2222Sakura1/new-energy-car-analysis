# -*- coding: utf-8 -*-
"""
新能源汽车产销与充电基础设施数据分析 —— Streamlit 交互式网站
图表与 Notebook《新能源汽车数据分析.ipynb》完全一致。
云端部署版：自带开源中文字体（思源黑体，SIL OFL 协议），数据路径基于本文件目录。
启动：streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ===================== 路径与中文字体（云端兼容关键）=====================
# BASE_DIR 取 app.py 所在目录，数据文件和字体都与 app.py 放在一起，云端本地都能读到
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, 'NotoSansSC-Regular.otf')

sns.set_style('whitegrid')
# 云端 Linux 默认没有 SimHei，这里加载仓库自带的思源黑体；本地也统一用它，保证图表一致
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)          # 把字体文件注册进 matplotlib
    _cn_font = fm.FontProperties(fname=FONT_PATH).get_name()  # 读出字体内部名称
    plt.rcParams['font.sans-serif'] = [_cn_font]
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示

st.set_page_config(page_title='新能源汽车数据分析平台', page_icon='🚗', layout='wide')

# ===================== 自定义样式 =====================
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
.banner {background: linear-gradient(120deg,#0E9F8C 0%,#14B8A6 55%,#2C7FB8 100%);
  padding: 30px 38px; border-radius: 16px; color:#fff; margin-bottom: 24px;
  box-shadow: 0 8px 22px rgba(14,159,140,.28);}
.banner h1 {margin:0; font-size:30px; font-weight:800; letter-spacing:.5px;}
.banner p {margin:10px 0 0; font-size:14px; opacity:.94;}
[data-testid="stMetric"] {background:#fff; border:1px solid #E6EFED; border-radius:12px;
  padding:14px 18px; box-shadow:0 2px 10px rgba(0,0,0,.05);}
[data-testid="stMetricValue"] {color:#0E7C6E; font-weight:800;}
section[data-testid="stSidebar"] {background:#F0F5F4;}
.concl {background:#F0F9F7; border-left:4px solid #14B8A6; padding:12px 18px;
  border-radius:0 10px 10px 0; font-size:14.5px; margin-top:10px; line-height:1.7;}
h1,h2,h3 {color:#16323A;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
  <h1>🚗 新能源汽车产销与充电基础设施数据分析平台</h1>
  <p>数据来源：和鲸社区《新能源汽车有关数据》　｜　分析流程：理解数据 → 数据清洗 → 特征工程 → 可视化分析 → 销量预测</p>
</div>
""", unsafe_allow_html=True)

F_BRAND = os.path.join(BASE_DIR, '1汽车分品牌产销.xlsx')
F_TOTAL = os.path.join(BASE_DIR, '3新能源汽车总体产销.xlsx')
F_CHARGE = os.path.join(BASE_DIR, '10国内充电设施数量.xls')
NATION_COL = '公共类充电桩数量_全国'

# ===================== 清洗函数（与 Notebook 一致）=====================
def power_type(name):
    s = str(name).upper().replace(' ', '')
    if ('PHEV' in s) or ('EREV' in s) or ('增程' in s): return '插电混动(PHEV)'
    if ('FCEV' in s) or ('氢' in s): return '燃料电池(FCEV)'
    if 'HEV' in s: return '油电混动(HEV)'
    if ('BEV' in s) or ('IEV' in s) or re.search(r'(?<![A-Z])EV(?![A-Z])', s): return '纯电动(BEV)'
    return '其他/燃油'

def clean_brand():
    d = pd.read_excel(F_BRAND, engine='openpyxl')
    d['数据日期'] = pd.to_datetime(d['数据日期'], errors='coerce')
    d = d.dropna(subset=['统计类型', '制造厂', '车型', '数据日期'])
    d = d.drop_duplicates(subset=['统计类型', '制造厂', '车型', '数据日期'])
    d = d.dropna(subset=['当期值(辆)'])
    d = d[d['当期值(辆)'] >= 0].copy()
    for c in ['统计类型', '制造厂', '车型', '车型大类']:
        d[c] = d[c].astype(str).str.strip()
    d['动力类型'] = d['车型'].apply(power_type)
    d['是否新能源'] = d['动力类型'].isin(['纯电动(BEV)', '插电混动(PHEV)', '燃料电池(FCEV)'])
    d['年份'] = d['数据日期'].dt.year
    d['年月'] = d['数据日期'].dt.to_period('M').astype(str)
    return d.reset_index(drop=True)

def clean_total():
    d = pd.read_excel(F_TOTAL, engine='openpyxl').iloc[1:].copy()
    d.columns = ['序号', '车型大类', '车型细分', '燃料类型', '数据日期',
                 '产量_当期', '产量_同比', '产量_累计', '产量_累计同比',
                 '销量_当期', '销量_同比', '销量_累计', '销量_累计同比']
    d['数据日期'] = pd.to_datetime(d['数据日期'], errors='coerce')
    for c in ['产量_当期', '销量_当期', '产量_累计', '销量_累计']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['数据日期'])
    d['产销率(%)'] = (d['销量_当期'] / d['产量_当期'] * 100).round(1)
    return d.reset_index(drop=True)

def clean_charge():
    raw = pd.read_excel(F_CHARGE, engine='xlrd', header=None)
    d = raw.iloc[6:].copy(); d.columns = raw.iloc[1].tolist()
    d = d.rename(columns={'指标名称': '月份'})
    d['月份'] = pd.to_datetime(d['月份'], errors='coerce')
    d = d.dropna(subset=['月份']).sort_values('月份').reset_index(drop=True)
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['全国充电桩同比(%)'] = d[NATION_COL].pct_change(12) * 100
    return d

@st.cache_data
def load_data():
    return clean_brand(), clean_total(), clean_charge()

brand_f, total_f, charge_f = load_data()

# 公共数据
tot = total_f[(total_f['车型大类'] == '总计') & (total_f['燃料类型'] == '总计')].sort_values('数据日期')
last_m = tot['数据日期'].max()

# ===================== 侧边导航 =====================
with st.sidebar:
    st.markdown('### 📂 功能导航')
    page = st.radio('导航', ['① 数据概览', '② 数据清洗', '③ 可视化分析', '④ 销量预测'],
                    label_visibility='collapsed')
    st.markdown('---')
    st.markdown('### 📊 数据规模')
    st.caption(f'分品牌明细：**{brand_f.shape[0]}** 行 × {brand_f.shape[1]} 列\n\n'
               f'总体产销：**{total_f.shape[0]}** 行 × {total_f.shape[1]} 列\n\n'
               f'充电设施：**{charge_f.shape[0]}** 个月 × {charge_f.shape[1]} 列')
    st.markdown('---')
    st.caption('时间范围：' + str(brand_f['数据日期'].min().date()) + ' ~ ' + str(last_m.date()))

# ===================== ① 数据概览 =====================
if page.startswith('①'):
    st.markdown('## 一、数据概览')
    c1, c2, c3 = st.columns(3)
    c1.metric('分品牌明细', f'{brand_f.shape[0]} 行', f'{brand_f.shape[1]} 列')
    c2.metric('总体产销表', f'{total_f.shape[0]} 行', f'{total_f.shape[1]} 列')
    c3.metric('充电设施表', f'{charge_f.shape[0]} 月', f'{charge_f.shape[1]} 列')
    c4, c5, c6, c7 = st.columns(4)
    c4.metric('制造厂数量', brand_f['制造厂'].nunique())
    c5.metric('车型数量', brand_f['车型'].nunique())
    c6.metric('起始月份', str(brand_f['数据日期'].min().date()))
    c7.metric('最新月份', str(last_m.date()))
    st.markdown('#### 🔎 数据表浏览')
    t1, t2, t3 = st.tabs(['分品牌产销明细', '总体产销', '充电设施'])
    with t1: st.dataframe(brand_f.head(200), use_container_width=True)
    with t2: st.dataframe(total_f.head(200), use_container_width=True)
    with t3: st.dataframe(charge_f.head(200), use_container_width=True)

# ===================== ② 数据清洗 =====================
elif page.startswith('②'):
    st.markdown('## 二、数据清洗')
    st.markdown('主表 `1汽车分品牌产销.xlsx`：原始 **30310 行 × 13 列** → 清洗后 **29984 行 × 17 列**')
    steps = pd.DataFrame({
        '步骤': ['① 删除空白行', '② 业务键去重', '③ 核心字段缺失', '④ 负值异常', '⑤ 文本清洗', '⑥ 特征衍生'],
        '处理数量': ['165 行', '46 行', '2 行', '113 行', '全表', '+4 新列'],
        '处理方式': ['删除', '去重保留第一条', '删除', '剔除', '去空格+正则提动力类型', '动力类型/是否新能源/年份/年月'],
        '原因': ['关键字段全空', '同车型同月重复录入', '当期值是分析基础', '数量不可能为负', '写法混乱需统一', '便于分组与时间分析'],
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)
    st.markdown('<div class="concl">缺失处理：当期同比缺 <b>16.96%</b>、环比缺 <b>12.57%</b>，'
                '属于"新车型无去年同期、首月无上期"的<b>业务缺失</b>，保留不填；'
                '文件3删除二级表头行、文件10跳过6行说明并重置表头。</div>', unsafe_allow_html=True)
    st.markdown('#### 清洗后主表（随机抽样 10 行）')
    st.dataframe(brand_f.sample(10, random_state=1), use_container_width=True)

# ===================== ③ 可视化分析（7 张图，与 Notebook 一致）=====================
elif page.startswith('③'):
    st.markdown('## 三、可视化分析（7 张图）')
    tabs = st.tabs(['图1 产销趋势+动力结构', '图2 TOP15品牌', '图3 纯电插混+车型',
                    '图4 充电桩+省份', '图5 车桩协同', '图6 产销散点', '图7 分布+箱线'])

    # ---------- 图1 ----------
    with tabs[0]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(tot['数据日期'], tot['产量_当期']/1e4, label='产量', color='#2C7FB8', lw=2)
        ax[0].plot(tot['数据日期'], tot['销量_当期']/1e4, label='销量', color='#F03B20', lw=2)
        ax[0].set_title('图1-a 全国新能源汽车月度产销量（折线图）'); ax[0].set_ylabel('万辆'); ax[0].legend()
        mix = total_f[(total_f['车型大类'] == '总计') & (total_f['数据日期'] == last_m)
                      & total_f['燃料类型'].isin(['纯电动', '插电式混合动力', '燃料电池'])].set_index('燃料类型')['销量_当期']
        mix = mix[mix > 0]
        ax[1].pie(mix.values, labels=[i.replace('插电式混合动力', '插电混动') for i in mix.index],
                  autopct='%1.1f%%', colors=['#41B6C4', '#FE9929', '#78C679'], startangle=90)
        ax[1].set_title('图1-b ' + str(last_m.date())[:7] + ' 销量动力结构（饼图）')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图1结论</b>：左图月度产销从 2019 年初几万辆增长到 2022 年月均 60~70 万辆，'
                    '长期向上、年末有冲高；右图 ' + str(last_m.date())[:7] + ' 销量中纯电动约 76%、'
                    '插电混动约 24%，技术路线以纯电为主。</div>', unsafe_allow_html=True)

    # ---------- 图2 ----------
    with tabs[1]:
        sales2022 = brand_f[(brand_f['统计类型'] == '销量') & brand_f['是否新能源'] & (brand_f['年份'] == 2022)]
        top = sales2022.groupby('制造厂')['当期值(辆)'].sum().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top.values/1e4, y=top.index, color='#2C7FB8', ax=ax)
        ax.set_title('图2 2022年（1-10月）新能源汽车销量 TOP15 制造厂（柱状图）')
        ax.set_xlabel('销量（万辆）'); ax.set_ylabel('')
        for i, v in enumerate(top.values): ax.text(v/1e4+0.5, i, f'{v/1e4:.1f}', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图2结论</b>：比亚迪以约 130.5 万辆断层领先，是第二名特斯拉(上海)的两倍多；'
                    'TOP5 为比亚迪、特斯拉(上海)、上汽通用五菱、吉利、广汽乘用车，头部集中明显。</div>',
                    unsafe_allow_html=True)

    # ---------- 图3 ----------
    with tabs[2]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for ft, col in [('纯电动', '#41B6C4'), ('插电式混合动力', '#FE9929')]:
            g = total_f[(total_f['车型大类'] == '总计') & (total_f['燃料类型'] == ft)].sort_values('数据日期')
            ax[0].plot(g['数据日期'], g['销量_当期']/1e4, label=ft.replace('插电式混合动力', '插电混动'),
                       color=col, lw=2)
        ax[0].set_title('图3-a 纯电动与插电混动月度销量（折线图）'); ax[0].set_ylabel('万辆'); ax[0].legend()
        seg = brand_f[(brand_f['统计类型'] == '销量') & brand_f['是否新能源']
                      & brand_f['车型大类'].isin(['轿车', 'SUV', 'MPV', '交叉型乘用车'])]
        seg_g = seg.groupby('车型大类', observed=True)['当期值(辆)'].sum().sort_values(ascending=False)
        sns.barplot(x=seg_g.index, y=seg_g.values/1e4, ax=ax[1], color='#78C679')
        ax[1].set_title('图3-b 各车型大类累计销量（柱状图）'); ax[1].set_ylabel('万辆'); ax[1].set_xlabel('')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图3结论</b>：纯电规模始终领先但插混 2021 年后增速更快；'
                    '车型以轿车、SUV 为绝对主力，MPV 和交叉型占比很小。</div>', unsafe_allow_html=True)

    # ---------- 图4 ----------
    with tabs[3]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(charge_f['月份'], charge_f[NATION_COL]/1e4, color='#2C7FB8', lw=2)
        ax[0].fill_between(charge_f['月份'], charge_f[NATION_COL]/1e4, alpha=0.2, color='#2C7FB8')
        ax[0].set_title('图4-a 全国公共类充电桩保有量（折线图）'); ax[0].set_ylabel('万个')
        prov = [c for c in charge_f.columns if str(c).startswith('公共类充电桩数量_')
                and c != NATION_COL and '交流' not in c and '直流' not in c]
        pr = charge_f.iloc[-1][prov].astype(float).sort_values(ascending=False).head(10)
        pr.index = [c.replace('公共类充电桩数量_', '') for c in pr.index]
        sns.barplot(x=pr.values/1e4, y=pr.index, ax=ax[1], color='#F03B20')
        ax[1].set_title('图4-b 重点省份充电桩保有量（柱状图，' + str(charge_f['月份'].max().date())[:7] + '）')
        ax[1].set_xlabel('万个'); ax[1].set_ylabel('')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图4结论</b>：全国公共充电桩从 2016 年初约 6 万个增至 2023 年初约 187 万个'
                    '（约 31 倍）；广东约 40 万个遥遥领先，江苏、浙江、上海、北京次之，区域向东部集中。</div>',
                    unsafe_allow_html=True)

    # ---------- 图5 ----------
    with tabs[4]:
        m = total_f[(total_f['车型大类'] == '总计') & (total_f['燃料类型'] == '总计')][['数据日期', '销量_当期']].copy()
        m['年月'] = m['数据日期'].dt.to_period('M').astype(str)
        c2 = charge_f[['月份', NATION_COL]].copy(); c2['年月'] = c2['月份'].dt.to_period('M').astype(str)
        mm = m.merge(c2, on='年月', how='inner')
        mm = mm[mm['数据日期'] >= '2019-01-01'].sort_values('数据日期')
        fig, ax1 = plt.subplots(figsize=(11, 4.5))
        ax1.bar(mm['数据日期'], mm['销量_当期']/1e4, color='#6BAED6', alpha=0.85, label='月度销量(万辆)')
        ax1.set_ylabel('新能源汽车月度销量（万辆）')
        ax2 = ax1.twinx()
        ax2.plot(mm['数据日期'], mm[NATION_COL]/1e4, color='#D7301F', lw=2, label='公共充电桩(万个)')
        ax2.set_ylabel('公共充电桩保有量（万个）', color='#D7301F')
        ax1.set_title('图5 新能源汽车销量与公共充电桩保有量协同增长（双轴组合图）')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图5结论</b>：汽车销量（柱）与充电桩保有量（线）同步快速上升，'
                    '产业与基础设施协同扩张，但充电桩区域分布不均。</div>', unsafe_allow_html=True)

    # ---------- 图6 ----------
    with tabs[5]:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(tot['产量_当期']/1e4, tot['销量_当期']/1e4, alpha=0.6, color='#2C7FB8')
        lim = [0, max(tot['产量_当期'].max(), tot['销量_当期'].max())/1e4*1.05]
        ax.plot(lim, lim, 'r--', lw=1.5, label='y=x 产销平衡线')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('月产量（万辆）'); ax.set_ylabel('月销量（万辆）')
        ax.set_title('图6 全国月产量与月销量关系（散点图）'); ax.legend()
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图6结论</b>：点整体紧贴红色对角线，说明产销基本匹配；'
                    '多数点略低于对角线（补库存），个别月份销量高于产量（去库存），没有明显滞销。</div>',
                    unsafe_allow_html=True)

    # ---------- 图7 ----------
    with tabs[6]:
        sales = brand_f[(brand_f['统计类型'] == '销量') & brand_f['是否新能源']
                        & brand_f['动力类型'].isin(['纯电动(BEV)', '插电混动(PHEV)'])
                        & (brand_f['当期值(辆)'] > 0)].copy()
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].hist(np.log10(sales['当期值(辆)']), bins=40, color='#41B6C4', edgecolor='white')
        ax[0].set_title('图7-a 车型月销量分布（直方图，log10刻度）')
        ax[0].set_xlabel('log10(月销量/辆)'); ax[0].set_ylabel('车型-月份记录数')
        sns.boxplot(data=sales, x='动力类型', y=np.log10(sales['当期值(辆)']),
                    ax=ax[1], hue='动力类型', legend=False, palette=['#41B6C4', '#FE9929'])
        ax[1].set_title('图7-b 纯电与插混月销量对比（箱线图，log10刻度）'); ax[1].set_ylabel('log10(月销量/辆)')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="concl"><b>图7结论</b>：月销量呈明显<b>右偏分布</b>（直方图）——多数车型月销几百到'
                    '几千辆，少数爆款月销数万辆，存在长尾；箱线图显示纯电、插混中位数接近，但纯电的上限（爆款）更高。</div>',
                    unsafe_allow_html=True)

# ===================== ④ 销量预测 =====================
else:
    st.markdown('## 四、全国新能源汽车销量预测')
    st.markdown('模型：**线性回归**，特征 = 时间序号（趋势）+ 月份独热（季节性）。')
    col_a, col_b = st.columns(2)
    n_future = col_a.slider('🔮 预测未来月数', 1, 24, 12)
    start = col_b.selectbox('📅 训练数据起始年份', [2018, 2019, 2020], index=1)

    ts = tot[tot['数据日期'] >= f'{start}-01-01'].reset_index(drop=True).copy()
    ts['t'] = np.arange(len(ts)); ts['月'] = ts['数据日期'].dt.month
    X = pd.concat([ts[['t']], pd.get_dummies(ts['月'], prefix='月', drop_first=True)], axis=1).astype(float)
    y = ts['销量_当期']
    model = LinearRegression().fit(X, y)
    ts['拟合'] = model.predict(X)

    r2 = r2_score(y, ts['拟合']); mae = mean_absolute_error(y, ts['拟合'])/1e4
    m1, m2, m3 = st.columns(3)
    m1.metric('R²（拟合优度）', f'{r2:.3f}')
    m2.metric('MAE（平均绝对误差）', f'{mae:.1f} 万辆')
    m3.metric('预测截止月份', str((last_m + pd.DateOffset(months=n_future)).date()))

    fd = pd.date_range(last_m + pd.DateOffset(months=1), periods=n_future, freq='ME')
    future = pd.DataFrame({'数据日期': fd})
    future['t'] = np.arange(len(ts), len(ts)+n_future); future['月'] = future['数据日期'].dt.month
    Xf = pd.concat([future[['t']], pd.get_dummies(future['月'], prefix='月', drop_first=True)], axis=1)
    Xf = Xf.reindex(columns=X.columns, fill_value=0).astype(float)
    future['预测销量(万辆)'] = np.clip((model.predict(Xf)/1e4).round(1), 0, None)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts['数据日期'], y/1e4, color='#2C7FB8', lw=2, label='历史销量')
    ax.plot(ts['数据日期'], ts['拟合']/1e4, color='#7FCDBB', lw=1.5, label='模型拟合')
    ax.plot(future['数据日期'], future['预测销量(万辆)'], color='#F03B20', lw=2, ls='--', label='预测销量')
    ax.set_ylabel('万辆'); ax.legend(); ax.set_title('图8 全国新能源汽车月度销量预测')
    plt.tight_layout(); st.pyplot(fig)

    st.markdown('#### 预测结果明细')
    st.dataframe(future, use_container_width=True, hide_index=True)
    st.download_button('📥 下载预测结果 CSV', future.to_csv(index=False).encode('utf-8-sig'),
                       '销量预测结果.csv', 'text/csv')
