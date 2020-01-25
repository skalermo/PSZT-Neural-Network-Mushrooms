import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


fig, ax = plt.subplots()
ax.set_ylabel('Square loss')
ax.set_xlabel('Iteration')


class Chart:
    @staticmethod
    def addToChart( data):
        ax.plot(data)

    @staticmethod
    def addLegend():
        pass

    @staticmethod
    def show():
        plt.show()

# # draw a tick corresponding to the firstGoodGeneration above the chart
# generationTick = [firstGoodGeneration]
# ax.xaxis.set_minor_locator(ticker.FixedLocator(generationTick))
# ax.xaxis.set_minor_formatter(ticker.FixedFormatter(generationTick))
# ax.tick_params(axis="x", which="minor", direction="out",
#                top=True, labeltop=True, bottom=False, labelbottom=False)
#
# # draw a tick corresponding to the the lowest achieved cost
# costTick = [progress[-1]]
# ax.yaxis.set_minor_locator(ticker.FixedLocator(costTick))
# ax.yaxis.set_minor_formatter(ticker.FixedFormatter(costTick))
# ax.tick_params(axis="y", which="minor", direction="out",
#                right=True, labelright=True, left=False, labelleft=False)
#
# ax.set_ylabel('Distance')
# ax.set_xlabel('Generation')
#
# # save chart
# fig.set_size_inches(18, 10, forward=True)
# fig.set_dpi(100)
# fig.savefig('chart.png', dpi=100)
#
# plt.show()
# plt.close()