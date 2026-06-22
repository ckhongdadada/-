"""生成20页幻灯片HTML，保持原有风格"""
import json, os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PROJECT = Path('C:/Users/28414/Desktop/机器学习—图像分类-期末汇报')
SLIDE_DIR = PROJECT / 'new_slides'
SLIDE_DIR.mkdir(exist_ok=True)

data = json.load(open(PROJECT / 'figures' / 'slide_data.json'))
slides = []

def g(key, default=0):
    return data.get(key, default)

# Hexagon decoration SVG
HEX_DECOR = '''<!-- Hexagon decorations -->
<div style="position:absolute;bottom:0;right:0;pointer-events:none;opacity:0.25">
<svg width="280" height="260" viewBox="0 0 280 260" fill="none" xmlns="http://www.w3.org/2000/svg">
<polygon points="140,10 245,50 245,130 140,170 35,130 35,50" stroke="#2563EB" stroke-width="1" fill="none"/>
<polygon points="140,50 210,75 210,125 140,150 70,125 70,75" stroke="#2563EB" stroke-width="1" fill="none"/>
<polygon points="140,90 175,105 175,135 140,150 105,135 105,105" stroke="#2563EB" stroke-width="1" fill="none"/>
<polygon points="210,90 245,105 245,135 210,150 175,135 175,105" stroke="#2563EB" stroke-width="0.5" fill="none" opacity="0.5"/>
<polygon points="70,90 105,105 105,135 70,150 35,135 35,105" stroke="#2563EB" stroke-width="0.5" fill="none" opacity="0.5"/>
</svg>
</div>'''

# Style template - use template substitution to avoid CSS brace conflicts
def get_style(bg, fg, card_bg, card_border, table_alt):
    return f'''<style>
@font-face{{font-family:'Alibaba PuHuiTi 3.0';src:url('file:///C%3A/Users/28414/.qoderwork/assets/ai-slides/fonts/AlibabaPuHuiTi-3-55-Regular.ttf') format('truetype');font-weight:400;font-style:normal;font-display:swap}}
@font-face{{font-family:'Alibaba PuHuiTi 3.0';src:url('file:///C%3A/Users/28414/.qoderwork/assets/ai-slides/fonts/AlibabaPuHuiTi-3-65-Medium.ttf') format('truetype');font-weight:500;font-style:normal;font-display:swap}}
@font-face{{font-family:'Alibaba PuHuiTi 3.0';src:url('file:///C%3A/Users/28414/.qoderwork/assets/ai-slides/fonts/AlibabaPuHuiTi-3-85-Bold.ttf') format('truetype');font-weight:700;font-style:normal;font-display:swap}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Alibaba PuHuiTi 3.0','Microsoft YaHei',sans-serif;background:{bg};color:{fg};width:1280px;height:720px;overflow:hidden;position:relative}}
.top-bar{{position:absolute;top:0;left:0;width:100%;height:2.5px;background:#2563EB;z-index:10}}
.slide-root{{position:absolute;top:2.5px;left:0;right:0;bottom:0;display:flex;flex-direction:column;padding:24px 56px 0 56px}}
.page-header{{flex-shrink:0;margin-bottom:12px}}
.header-label{{font-size:18px;font-weight:500;color:#2563EB;text-transform:uppercase;letter-spacing:0.13em;line-height:1.2;margin-bottom:8px}}
.header-divider{{width:100%;height:0;border:none;border-top:0.5px solid #E2E8F0;margin-bottom:8px}}
.header-title{{font-size:22px;font-weight:500;color:{fg};line-height:1.3}}
.main-content{{flex:1;display:flex;gap:28px;min-height:0;padding-bottom:10px}}
.card{{background:{card_bg};border:1px solid {card_border};border-radius:12px;padding:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#2563EB;color:#fff;padding:8px 10px;text-align:left;font-weight:500;font-size:12px}}
td{{padding:6px 10px;border-bottom:1px solid #E2E8F0;font-size:12px}}
tr:nth-child(even) td{{background:{table_alt}}}
.highlight{{background:#FEF3C7;font-weight:600}}
</style>'''

def make_slide(num, title, label, content, dark=False):
    bg = '#0F172A' if dark else '#FFFFFF'
    fg = '#FFFFFF' if dark else '#0F172A'
    card_bg = 'rgba(255,255,255,0.06)' if dark else '#F8FAFC'
    card_border = 'rgba(255,255,255,0.12)' if dark else '#E2E8F0'
    table_alt = 'rgba(255,255,255,0.03)' if dark else '#F8FAFC'
    style = get_style(bg, fg, card_bg, card_border, table_alt)
    header = f'''<div class="page-header">
<div class="header-label">{label}</div>
<hr class="header-divider">
<div class="header-title">{title}</div>
</div>'''
    html = f'''<!-- ai-slides: slot=slot-{num:02d} -->
<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
{style}</head><body>
<div class="top-bar"></div>
{HEX_DECOR if not dark else ""}
<div class="slide-root">{header}<div class="main-content">{content}</div></div>
</body></html>'''
    path = SLIDE_DIR / f'{num}.html'
    path.write_text(html, encoding='utf-8')
    slides.append(path)
    return path

# ===== Slide 1: Title (dark) =====
make_slide(1, '低标注预算下长尾分布的主动学习与半监督学习联合策略研究', '', '''
<div style="display:flex;align-items:center;justify-content:center;flex:1">
<div style="text-align:center;max-width:900px">
<p style="font-size:18px;color:#94A3B8;margin-bottom:32px">从均衡到极端长尾的完整分布谱系实验 · 尾类感知AL · 联合分布感知 · 渐进式SSL</p>
<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
<span style="padding:6px 16px;border-radius:20px;font-size:13px;background:#2563EB;color:#fff">CIFAR-10 / CIFAR-100</span>
<span style="padding:6px 16px;border-radius:20px;font-size:13px;background:#10B981;color:#fff">SimpleCNN + ResNet-18</span>
<span style="padding:6px 16px;border-radius:20px;font-size:13px;background:#F59E0B;color:#0F172A">ρ ∈ {1,5,10,20,50,100}</span>
</div></div></div>
''', dark=True)

# ===== Slide 2: Outline =====
cards = [
    ('01 研究背景', '长尾分布 + 低标注成本的核心矛盾'),
    ('02 实验设置', '数据构造、模型架构、评估指标'),
    ('03 标准AL策略', '6种策略在分布谱系上的表现'),
    ('04 尾类感知AL', 'Class-Aware / Gap-Aware 策略设计'),
    ('05 AL+SSL联合策略', '联合分布感知 + 渐进式调度'),
    ('06 对比与消融', 'LDAM、CB/Focal、ResNet-18、TML'),
    ('07 失效分析', '创新策略的适用边界'),
    ('08 总结展望', '核心贡献与未来方向'),
]
cards_html = ''.join(f'<div class="card"><h3 style="font-size:15px;font-weight:600;color:#2563EB;margin-bottom:8px">{t}</h3><p style="font-size:13px;color:#64748B">{d}</p></div>' for t,d in cards)
make_slide(2, '汇报目录', 'CONTENTS', f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;flex:1">{cards_html}</div>')

# ===== Slide 3: Core Problem =====
make_slide(3, '核心矛盾与研究问题', 'MOTIVATION', '''
<div style="display:flex;gap:20px;flex:1">
<div class="card" style="flex:1;border-left:4px solid #DC2626">
<h3 style="color:#DC2626;font-size:15px;margin-bottom:12px">标注成本高昂</h3>
<p style="font-size:13px;color:#475569;line-height:1.6">医疗影像、工业缺陷检测等领域，专家标注成本极高。CIFAR-10的50K训练图像中仅标注1100个（2.2%）。</p>
</div>
<div class="card" style="flex:1;border-left:4px solid #2563EB">
<h3 style="color:#2563EB;font-size:15px;margin-bottom:12px">三大研究问题</h3>
<p style="font-size:13px;color:#475569;line-height:1.8"><b>RQ1</b>：标准AL策略在长尾下是否失效？<br><b>RQ2</b>：如何设计尾类感知的AL策略？<br><b>RQ3</b>：AL+SSL如何协同？</p>
</div>
<div class="card" style="flex:1;border-left:4px solid #059669">
<h3 style="color:#059669;font-size:15px;margin-bottom:12px">本文方案</h3>
<p style="font-size:13px;color:#475569;line-height:1.8">① 尾类感知AL策略<br>② 联合分布感知<br>③ 渐进式SSL调度</p>
</div>
</div>
''')

# ===== Slide 4: Standard AL Results =====
std = {rho: g(f'std_rho{rho}') for rho in [1,5,10,20,50,100]}
fs = {rho: g(f'fs_rho{rho}') for rho in [1,5,10,20,50,100]}
rows = ''
for s in ['random','entropy','margin','coreset','badge','qbc']:
    cells = ''.join(f'<td>{std[rho].get(s,0):.4f}</td>' for rho in [1,5,10,20,50,100])
    rows += f'<tr><td style="font-weight:600">{s.title()}</td>{cells}</tr>'
fs_cells = ''.join(f'<td style="font-weight:700;color:#2563EB">{fs[rho]:.4f}</td>' for rho in [1,5,10,20,50,100])
rows += f'<tr><td style="font-weight:600">Full Supervised</td>{fs_cells}</tr>'
make_slide(4, '标准AL策略在分布谱系上的表现', 'EXPERIMENT', f'''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table><tr><th>策略</th><th>ρ=1</th><th>ρ=5</th><th>ρ=10</th><th>ρ=20</th><th>ρ=50</th><th>ρ=100</th></tr>{rows}</table>
<div style="display:flex;gap:16px;margin-top:auto">
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#DC2626">-50%</div><div style="font-size:12px;color:#64748B">ρ=1→100 F1下降</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#2563EB">QBC</div><div style="font-size:12px;color:#64748B">ρ=10最优策略</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">Margin</div><div style="font-size:12px;color:#64748B">ρ=50/100最优</div></div>
</div></div>
''')

# ===== Slide 5: Innovative AL+SSL Results =====
innov = {rho: g(f'innov_rho{rho}') for rho in [1,10,50,100]}
rows2 = ''
for s in ['class_aware_entropy','gap_aware_entropy','adaptive_gap_entropy']:
    cells = ''.join(f'<td>{innov[rho].get(s,0):.4f}</td>' for rho in [1,10,50,100])
    name = s.replace('_entropy','').replace('_',' ').title()
    rows2 += f'<tr><td style="font-weight:600">{name}</td>{cells}</tr>'
make_slide(5, '尾类感知AL+SSL策略结果', 'INNOVATION', f'''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table><tr><th>策略</th><th>ρ=1</th><th>ρ=10</th><th>ρ=50</th><th>ρ=100</th></tr>{rows2}</table>
<div style="display:flex;gap:16px;margin-top:auto">
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+21.0%</div><div style="font-size:12px;color:#64748B">ρ=50 Class-Aware vs Entropy</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+7.3%</div><div style="font-size:12px;color:#64748B">ρ=100 Class-Aware vs Entropy</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#2563EB">p&lt;0.05</div><div style="font-size:12px;color:#64748B">统计显著</div></div>
</div></div>
''')

# ===== Slide 6: Joint Distribution =====
pj = {}
for cfg in ['labeled_only','joint_r0','joint_r3','joint_r5','joint_r7']:
    pj[cfg] = {}
    for rho in [10,50,100]:
        d = g(f'pj_{cfg}_rho{rho}')
        if isinstance(d, dict):
            pj[cfg][rho] = d
        else:
            pj[cfg][rho] = {}

cfg_names = {'labeled_only':'纯AL','joint_r0':'全程联合','joint_r3':'渐进r3','joint_r5':'渐进r5','joint_r7':'渐进r7'}
rows3 = ''
for cfg in ['labeled_only','joint_r0','joint_r3','joint_r5','joint_r7']:
    cells = ''
    for rho in [10,50,100]:
        vals = pj[cfg].get(rho, {})
        best_s = max(vals, key=vals.get) if vals else ''
        best_v = vals.get(best_s, 0)
        cells += f'<td>{best_v:.4f}</td>'
    rows3 += f'<tr><td style="font-weight:600">{cfg_names[cfg]}</td>{cells}</tr>'

make_slide(6, '渐进式联合分布感知实验', 'JOINT DISTRIBUTION', f'''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<p style="font-size:13px;color:#475569">AL查询时将模型预测的伪标签纳入类分布计算，使AL感知SSL已覆盖的类别。渐进式：早期纯AL，后期引入联合分布。</p>
<table><tr><th>配置</th><th>ρ=10 (最优策略)</th><th>ρ=50</th><th>ρ=100</th></tr>{rows3}</table>
<div style="display:flex;gap:16px;margin-top:auto">
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+3.9%</div><div style="font-size:12px;color:#64748B">ρ=10 联合 vs 纯AL</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+9.0%</div><div style="font-size:12px;color:#64748B">ρ=50 联合 vs 纯AL</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+9.8%</div><div style="font-size:12px;color:#64748B">ρ=100 渐进 vs 纯AL</div></div>
</div></div>
''')

# ===== Slide 7: Ablation =====
make_slide(7, '消融实验：AL创新 vs SSL创新', 'ABLATION', '''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table>
<tr><th>配置</th><th>AL策略</th><th>SSL方法</th><th>ρ=10 F1</th><th>vs基线</th></tr>
<tr><td>基线</td><td>Entropy</td><td>无</td><td>0.3438</td><td>—</td></tr>
<tr><td>AL+Base SSL</td><td>Entropy</td><td>FlexMatch</td><td>0.3639</td><td>+5.8%</td></tr>
<tr class="highlight"><td><b>Innov AL+Base SSL</b></td><td><b>Class-Aware</b></td><td>FlexMatch</td><td><b>0.4017</b></td><td><b>+16.8%</b></td></tr>
<tr><td>AL+Innov SSL</td><td>Entropy</td><td>Deficit+加权</td><td>0.3351</td><td>-2.5%</td></tr>
<tr><td>Innov AL+Innov SSL</td><td>Class-Aware</td><td>Deficit+加权</td><td>0.3729</td><td>+8.5%</td></tr>
</table>
<div style="margin-top:auto;padding:16px;background:#F0F9FF;border-radius:8px;border-left:4px solid #2563EB">
<p style="font-size:14px;color:#1E40AF"><b>核心发现</b>：AL创新是主要贡献来源（+16.8%），SSL创新单独使用反而降F1（-2.5%）。联合效果不如AL创新+基础SSL。</p>
</div></div>
''')

# ===== Slide 8: Progressive SSL =====
make_slide(8, '渐进式SSL调度策略', 'PROGRESSIVE SSL', '''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<p style="font-size:13px;color:#475569">前期使用Base SSL（固定τ=0.95）保证伪标签质量，后期切换至Innov SSL引入类别感知。</p>
<table>
<tr><th>SSL方法</th><th>ρ=1</th><th>ρ=10</th><th>ρ=50</th><th>ρ=100</th></tr>
<tr><td>无SSL</td><td>0.4217</td><td>0.3573</td><td>0.2860</td><td>0.2571</td></tr>
<tr><td>Base SSL</td><td>0.4339</td><td><b>0.4027</b></td><td>0.3258</td><td>0.2844</td></tr>
<tr><td>Innov SSL</td><td>0.4213</td><td>0.3590</td><td>0.3220</td><td>0.2856</td></tr>
<tr class="highlight"><td><b>Progressive r3</b></td><td>0.4422</td><td>0.3776</td><td><b>0.3322</b></td><td>0.2847</td></tr>
<tr><td>Progressive r5</td><td>0.4208</td><td>0.3838</td><td>0.3228</td><td><b>0.2988</b></td></tr>
</table>
<div style="margin-top:auto;padding:16px;background:#F0FDF4;border-radius:8px;border-left:4px solid #059669">
<p style="font-size:14px;color:#166534"><b>规律</b>：ρ越大，渐进式SSL越有效。ρ=100时Progressive r5比Base SSL提升+5.1%。</p>
</div></div>
''')

# ===== Slide 9: Failure Analysis =====
make_slide(9, '创新策略失效场景分析', 'FAILURE ANALYSIS', '''
<div style="flex:1;display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div class="card" style="border-left:4px solid #DC2626">
<h3 style="color:#DC2626;font-size:14px;margin-bottom:8px">均衡数据 (ρ=1)</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">类别惩罚项趋同，不提供有效区分信号。创新策略比标准Entropy低3-10%。</p>
</div>
<div class="card" style="border-left:4px solid #F59E0B">
<h3 style="color:#D97706;font-size:14px;margin-bottom:8px">极端长尾 (ρ=100)</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">尾类样本极少（<100），类分布估计不稳定。但渐进式联合分布仍有+9.8%提升。</p>
</div>
<div class="card" style="border-left:4px solid #F59E0B">
<h3 style="color:#D97706;font-size:14px;margin-bottom:8px">CIFAR-100 (100类)</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">每类仅~10样本，处于极度稀疏样本域。AL策略差异消失，需要新方法。</p>
</div>
<div class="card" style="border-left:4px solid #059669">
<h3 style="color:#059669;font-size:14px;margin-bottom:8px">适用条件</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">① 类间分布差异足够大（ρ≥10）<br>② 类内样本足够多<br>③ 类别数适中（10类优于100类）</p>
</div>
</div>
''')

# ===== Slide 10: LDAM + CB/Focal =====
make_slide(10, '对比实验：LDAM与CB/Focal基线', 'BASELINES', '''
<div style="flex:1;display:flex;gap:20px">
<div style="flex:1">
<h3 style="font-size:14px;font-weight:600;margin-bottom:8px">LDAM-DRW 全监督基线</h3>
<table>
<tr><th>方法</th><th>ρ=10</th><th>ρ=50</th><th>ρ=100</th></tr>
<tr><td>CE Full</td><td>0.7341</td><td>0.5831</td><td>0.5114</td></tr>
<tr class="highlight"><td><b>LDAM Full</b></td><td><b>0.7365</b></td><td><b>0.6049</b></td><td><b>0.5401</b></td></tr>
</table>
<p style="font-size:11px;color:#64748B;margin-top:8px">LDAM在全量数据上优于CE(+3.7%~5.6%)，但需全量标注，不适用于AL场景。</p>
</div>
<div style="flex:1">
<h3 style="font-size:14px;font-weight:600;margin-bottom:8px">CB/Focal Loss基线</h3>
<table>
<tr><th>方法</th><th>ρ=10</th><th>ρ=50</th><th>ρ=100</th></tr>
<tr><td>CE (Entropy)</td><td>0.3438</td><td>0.2585</td><td>0.2579</td></tr>
<tr><td>CB Loss</td><td>0.3693</td><td><b>0.3347</b></td><td><b>0.3084</b></td></tr>
<tr><td>Focal Loss</td><td><b>0.4000</b></td><td>0.2977</td><td>0.2391</td></tr>
</table>
<p style="font-size:11px;color:#64748B;margin-top:8px">CB在极端长尾最优，Focal在中等不平衡最优。</p>
</div>
</div>
''')

# ===== Slide 11: ResNet-18 =====
resnet = {rho: g(f'resnet_rho{rho}') for rho in [1,5,10,20,50,100]}
resnet_fs = {rho: g(f'resnet_fs_rho{rho}') for rho in [1,5,10,20,50,100]}
rows4 = ''
for s in ['random','entropy','margin','class_aware_entropy','gap_aware_entropy','adaptive_gap_entropy']:
    cells = ''.join(f'<td>{resnet[rho].get(s,0):.4f}</td>' for rho in [1,5,10,20,50,100])
    name = s.replace('_entropy','').replace('_',' ').title()
    rows4 += f'<tr><td style="font-weight:600">{name}</td>{cells}</tr>'
fs_cells = ''.join(f'<td style="font-weight:700;color:#2563EB">{resnet_fs[rho]:.4f}</td>' for rho in [1,5,10,20,50,100])
rows4 += f'<tr><td style="font-weight:600">Full Supervised</td>{fs_cells}</tr>'
make_slide(11, 'ResNet-18 深层网络验证', 'RESNET-18', f'''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table><tr><th>策略</th><th>ρ=1</th><th>ρ=5</th><th>ρ=10</th><th>ρ=20</th><th>ρ=50</th><th>ρ=100</th></tr>{rows4}</table>
<div style="display:flex;gap:16px;margin-top:auto">
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#059669">+83%</div><div style="font-size:12px;color:#64748B">ResNet vs SimpleCNN (ρ=50 Entropy)</div></div>
<div class="card" style="flex:1;padding:12px;text-align:center"><div style="font-size:24px;font-weight:700;color:#2563EB">0.4807</div><div style="font-size:12px;color:#64748B">ρ=50 Adaptive Gap最优</div></div>
</div></div>
''')

# ===== Slide 12: CIFAR-100 =====
make_slide(12, 'CIFAR-100 交叉验证', 'CROSS-DATASET', '''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table>
<tr><th>策略</th><th>ρ=1</th><th>ρ=10</th><th>ρ=50</th></tr>
<tr><td>Random (标准AL)</td><td>0.0888</td><td>0.0756</td><td>0.0531</td></tr>
<tr><td>Entropy (标准AL)</td><td>0.0634</td><td>0.0787</td><td>0.0474</td></tr>
<tr><td>Margin (标准AL)</td><td>0.0943</td><td>0.0851</td><td>0.0620</td></tr>
<tr><td>Class-Aware (创新)</td><td>0.0696</td><td>0.0687</td><td>0.0526</td></tr>
<tr><td>Gap-Aware (创新)</td><td>0.0633</td><td>0.0596</td><td>0.0505</td></tr>
<tr><td>Full Supervised</td><td>0.5652</td><td>0.4194</td><td>0.2829</td></tr>
</table>
<div style="margin-top:auto;padding:16px;background:#FEF3C7;border-radius:8px;border-left:4px solid #F59E0B">
<p style="font-size:14px;color:#92400E"><b>结论</b>：100类+1000标注=每类仅~10样本，处于极度稀疏样本域。AL策略差异消失，需要从输出空间转向特征空间的分布估计。</p>
</div></div>
''')

# ===== Slide 13: Per-Class F1 =====
make_slide(13, 'Per-Class F1 逐类分析', 'PER-CLASS', '''
<div style="flex:1;display:flex;gap:20px">
<div style="flex:1">
<h3 style="font-size:14px;font-weight:600;margin-bottom:8px">ρ=50 Head vs Tail</h3>
<table>
<tr><th>策略</th><th>Macro</th><th>Head(0-4)</th><th>Tail(5-9)</th></tr>
<tr><td>Entropy</td><td>0.2528</td><td>0.191</td><td>0.315</td></tr>
<tr><td>Class-Aware</td><td>0.3134</td><td>0.238</td><td>0.389</td></tr>
<tr class="highlight"><td><b>CA_ssl(r5)</b></td><td><b>0.3327</b></td><td><b>0.243</b></td><td><b>0.423</b></td></tr>
</table>
<p style="font-size:11px;color:#64748B;margin-top:8px">Class-Aware显著提升head类性能(+27.2%)</p>
</div>
<div style="flex:1">
<h3 style="font-size:14px;font-weight:600;margin-bottom:8px">ρ=100 Head vs Tail</h3>
<table>
<tr><th>策略</th><th>Macro</th><th>Head(0-4)</th><th>Tail(5-9)</th></tr>
<tr><td>Entropy</td><td>0.2760</td><td>0.241</td><td>0.311</td></tr>
<tr><td>Class-Aware</td><td>0.2635</td><td>0.220</td><td>0.307</td></tr>
<tr><td>CA_ssl(r5)</td><td>0.2646</td><td>0.209</td><td>0.320</td></tr>
</table>
<p style="font-size:11px;color:#64748B;margin-top:8px">ρ=100时创新策略损害head类，尾类样本太少导致惩罚噪声过大</p>
</div>
</div>
''')

# ===== Slide 14: TML =====
make_slide(14, 'TML vs DL 通用性对比', 'TML', '''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<table>
<tr><th>模型</th><th>ρ=1最优</th><th>ρ=10最优</th><th>ρ=50最优</th><th>策略分化</th></tr>
<tr><td>LR</td><td>0.2690</td><td>0.2078</td><td>0.1591</td><td>&lt;3%</td></tr>
<tr><td>RF</td><td>0.3357</td><td>0.2585</td><td>0.1689</td><td>~5%</td></tr>
<tr class="highlight"><td><b>SimpleCNN+SSL</b></td><td><b>0.4504</b></td><td><b>0.3939</b></td><td><b>0.3327</b></td><td><b>&gt;10%</b></td></tr>
<tr><td>ResNet-18</td><td>0.5686</td><td>0.5442</td><td>0.4807</td><td>&gt;10%</td></tr>
</table>
<div style="margin-top:auto;padding:16px;background:#F0F9FF;border-radius:8px;border-left:4px solid #2563EB">
<p style="font-size:14px;color:#1E40AF"><b>结论</b>：TML模型在原始像素上策略分化极小（&lt;5%），DL模型分化显著（&gt;10%）。AL策略的价值依赖于模型的特征提取能力。</p>
</div></div>
''')

# ===== Slide 15: Summary =====
make_slide(15, '总结与展望', 'CONCLUSION', '''
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div class="card" style="border-left:4px solid #2563EB">
<h3 style="color:#2563EB;font-size:14px;margin-bottom:8px">核心贡献</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">① 尾类感知AL策略：ρ=50时+21%<br>② 联合分布感知：ρ=50时+9%，无额外超参数<br>③ 渐进式联合分布：ρ=100时+9.8%</p>
</div>
<div class="card" style="border-left:4px solid #059669">
<h3 style="color:#059669;font-size:14px;margin-bottom:8px">关键发现</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">① 标准AL在长尾下效果有限<br>② AL创新是主要贡献来源<br>③ 创新策略在ρ≤50时有效</p>
</div>
<div class="card" style="border-left:4px solid #F59E0B">
<h3 style="color:#D97706;font-size:14px;margin-bottom:8px">局限性</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">① ρ=100时效果有限<br>② CIFAR-100上优势不明显<br>③ TML模型上策略分化小</p>
</div>
<div class="card" style="border-left:4px solid #8B5CF6">
<h3 style="color:#7C3AED;font-size:14px;margin-bottom:8px">未来方向</h3>
<p style="font-size:12px;color:#475569;line-height:1.6">① 预训练特征+AL策略<br>② 特征空间分布估计<br>③ 层级类别聚类</p>
</div>
</div></div>
''')

print(f"Generated {len(slides)} slides in {SLIDE_DIR}")