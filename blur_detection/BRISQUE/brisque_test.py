from brisque import BRISQUE

URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"

obj = BRISQUE(url=True)
obj.score(URL)