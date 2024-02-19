from snapshot_selenium import snapshot as driver
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
from pyecharts.globals import ThemeType
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = ""

# 数据
data = {
    "ISKG-a": [0.11, 0.10],
    "ISKG-s": [0.08, 0.09],
    "ISKG-K": [0.09, 0.08],
    "ISKG": [0.13, 0.13],
}

# 横坐标
x_axis = ["NDCG@20", "Recall@20"]

# 创建柱状图对象
bar = (
    Bar()
    .add_xaxis(x_axis)
    .add_yaxis("ISKG-a", data["ISKG-a"], color="#B6CB58")  # 自定义颜色
    .add_yaxis("ISKG-s", data["ISKG-s"], color="#5A6F19")  # 自定义颜色
    .add_yaxis("ISKG-K", data["ISKG-K"], color="#F36E2D")  # 自定义颜色
    .add_yaxis("ISKG", data["ISKG"], color="#000000")  # 自定义颜色
    .set_global_opts(

        xaxis_opts=opts.AxisOpts(
            type_="category",
            name_gap=100,
            splitline_opts=opts.SplitLineOpts(is_show=False),  # 取消横坐标框线
            axisline_opts=opts.AxisLineOpts(is_show=True,
                                            linestyle_opts=opts.LineStyleOpts(width=3)
                                            # 设置轴线宽度
                                            ),
            axislabel_opts=opts.LabelOpts(font_size=15, font_weight="bold")  # 设置坐标轴上的值的字体样式
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            min_=0,
            max_=0.2,
            name_gap=100,  # 坐标轴与坐标轴名间距
            splitline_opts=opts.SplitLineOpts(is_show=False),  # 取消纵坐标框线
            axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(width=3)
                                            # 设置轴线宽度
                                            ),
            axislabel_opts=opts.LabelOpts(font_size=15, font_weight="bold")  # 设置坐标轴上的值的字体样式
        ),
        toolbox_opts=opts.ToolboxOpts(is_show=True),  # 显示工具箱
        legend_opts=opts.LegendOpts(orient="vertical", pos_top="10%", pos_left="80%",
                                    textstyle_opts=opts.TextStyleOpts(font_weight="bold"))  # 图例位置
    )
)
# 渲染图表
bar.render("bar_rq2.html")
make_snapshot(driver, bar.render(), "bar_rq2.png")
