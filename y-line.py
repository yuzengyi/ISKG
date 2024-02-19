from snapshot_selenium import snapshot as driver
from pyecharts import options as opts
from pyecharts.charts import Line, Grid
from pyecharts.render import make_snapshot
from pyecharts.globals import SymbolType
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = ""
# 创建折线图对象
line = Line()

# 添加 x 轴数据
x_data = [i / 10 for i in range(11)]  # lambda 取值范围为 0 到 1，共 11 个点
line.add_xaxis(x_data)

# 添加Y轴数据
y_data_recall = [round(val, 2) for val in [0.12000000476837158, 0.12000000476837158, 0.12999999523162842, 0.14000000059604645, 0.15000000596046448, 0.10000000149011612, 0.07000000029802322, 0.10000000149011612, 0.12000000476837158, 0.07999999821186066, 0.10999999940395355]]
y_data_ndcg = [round(val, 2) for val in [0.08927138080318536, 0.12529251191053598, 0.06838886876459246, 0.08642320753440594, 0.07661168786440312, 0.14546048435863557, 0.1155319271097414, 0.09853034333300649, 0.09453977895121397, 0.0706863420635824, 0.12926152840930147]]

# 添加折线
line.add_yaxis("Recall@20", y_data_recall, symbol=SymbolType.TRIANGLE, symbol_size=10,label_opts=opts.LabelOpts(is_show=True,position='bottom'))
line.add_yaxis("NDCG@20", y_data_ndcg, symbol=SymbolType.ROUND_RECT, symbol_size=10,label_opts=opts.LabelOpts(is_show=True))

line.extend_axis()
line.extend_axis()
# 设置全局配置项
line.set_global_opts(

    xaxis_opts=opts.AxisOpts(name="λ", name_location="center",
                             name_textstyle_opts=opts.TextStyleOpts(font_size=25, font_weight="bold"), type_="value",
                             min_=0, max_=1, splitline_opts=opts.SplitLineOpts(is_show=False), name_gap=40,
                             axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(width=3)
                                                             # 设置轴线宽度
                                                             ),
                             axislabel_opts=opts.LabelOpts(font_size=15, font_weight="bold")  # 设置坐标轴上的值的字体样式
                             ),
    yaxis_opts=opts.AxisOpts(name="@20", name_textstyle_opts=opts.TextStyleOpts(font_size=25, font_weight="bold"),
                             name_location="center", type_="value", min_=0, max_=0.2, name_gap=40,
                             axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(width=3)
                                                             # 设置轴线宽度
                                                             ),
                             axislabel_opts=opts.LabelOpts(font_size=15, font_weight="bold")  # 设置坐标轴上的值的字体样式

                             ),
    legend_opts=opts.LegendOpts(
        pos_top="10%",  # 设置图例位置为图片内的顶部
        pos_left="75%",  # 设置图例位置为图片内的右侧
        orient="vertical",  # 设置图例为垂直方向
        textstyle_opts=opts.TextStyleOpts(font_weight="bold"),  # 将文字加粗

    )
)
# 设置线条样式
# line.set_series_opts(
#     linestyle_opts=opts.LineStyleOpts(width=2),
#     label_opts=opts.LabelOpts(is_show=True,position='bottom',color="black")
#
# )

# 渲染图表
line.render("ndcg_line_chart.html")
make_snapshot(driver, line.render(), "line_chart.png")#生成图片

# 创建组合图
# (Grid(init_opts=opts.InitOpts(width='750px', height='350px'))
#  .add(line, grid_opts=opts.GridOpts(pos_left="55%"))
#  .add(line, grid_opts=opts.GridOpts(pos_right="55%"))
#  ).render("grid_line.html")
