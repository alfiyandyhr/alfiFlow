# ==========================
# 		Import modules		
# ==========================

from alfiFlow import vonNeumannStabilityAnalysis, plot_stability
import numpy as np

# ==============================================
# 		Performing von Neumann analysis 		
# ==============================================

analyses = [vonNeumannStabilityAnalysis(jmax=101) for i in range(5)]
cfl_list = [0.25, 0.50, 0.75, 1.00, 1.25]

for i, analysis in enumerate(analyses):
	# analysis.do(cfl=cfl_list[i], scheme="CentralDifference")
	# analysis.do(cfl=cfl_list[i], scheme="FirstOrderUpwindDifference")
	analysis.do(cfl=cfl_list[i], scheme="LaxScheme")
	# analysis.do(cfl=cfl_list[i], scheme="LaxWendroffScheme")
	# analysis.do(cfl=cfl_list[i], scheme="SecondOrderUpwindDifference")
	# analysis.do(cfl=cfl_list[i], scheme="SecondOrderUpwindDifferenceModified")

plot_stability(analyses=analyses, type="amplification")

plot_stability(analyses=analyses, type="phase")