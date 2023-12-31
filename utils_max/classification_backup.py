regions_blank = {'none': 0, 'ausserhalb': 0, 'mund': 0, 'kehlkopf': 0, 'oseophagus': 0, 'z_linie': 0, 'magen': 0,
                 'pylorus': 0, 'duodenum': 0, 'inversion': 0}
n = list(range(0, 5336))
annot = {
    "Real-20200101-0002-0001_2": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': n[4932:4992],
        'z_linie': n[4012:4128] + n[4354:4441],
        'magen': n[4118:4929],
        'pylorus': [],
        'duodenum': [],
        'inversion': n[4360:4600]
    },
    "Real-20200821-0003-0001_1": {
        'ausserhalb': [],
        'mund': n[1089:1173],
        'kehlkopf': n[1148:1282],
        'oseophagus': n[1148:3027],
        'z_linie': n[1676:3027] + n[4354:4441],
        'magen': n[3027:4929],
        'pylorus': n[4788:5335],
        'duodenum': [],
        'inversion': []
    },
    "Real-20200821-0003-0001_2": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': [],
        'z_linie': [],
        'magen': n[3686:4652] + n[5109:5332],
        'pylorus': n[3986:4652] + n[5075:5109],
        'duodenum': n[18:3686] + n[4652:5075],
        'inversion': n[5216:5332]
    },
    "Real-20200821-0003-0001_3": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': n[2677:2744] + n[5092:5335],
        'z_linie': n[2677:5092],
        'magen': n[77:2677],
        'pylorus': n[821:1340],
        'duodenum': [],
        'inversion': n[77:587]
    },
    "Real-20200821-0003-0001_4": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': n[0:1029],
        'z_linie': [],
        'magen': [],
        'pylorus': [],
        'duodenum': [],
        'inversion': []
    },
    "Real-20200821-0004-0001_1": {
        'ausserhalb': n[2502:3607] + n[3859:5075],
        'mund': n[2502:3693] + n[5075:5132],
        'kehlkopf': n[2412:2502] + n[3750:3859] + n[5170:2535],
        'oseophagus': [],
        'z_linie': [],
        'magen': [],
        'pylorus': [],
        'duodenum': [],
        'inversion': []
    },
    "Real-20200821-0004-0001_2": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': n[0:618],
        'z_linie': n[618:1183],
        'magen': n[1183:3000] + n[4890:5139],
        'pylorus': n[2580:3000] + n[4164:4300],
        'duodenum': n[3000:4161],
        'inversion': n[4300:4890] + n[5068:5139]
    },
    "Real-20200821-0004-0001_3": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': [],
        'z_linie': n[5066:5335],
        'magen': n[0:5066],
        'pylorus': n[511:770],
        'duodenum': [],
        'inversion': n[3330:3473] + n[4160:4252]
    },
    "Real-20200821-0008-0001_1": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': n[1026:1220],
        'oseophagus': n[1220:1893],
        'z_linie': n[1893:1987],
        'magen': n[2387:4652],
        'pylorus': n[3470:4652],
        'duodenum': n[4686:5335],
        'inversion': []
    },
    "Real-20200821-0010-0001_1": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': n[738:923],
        'oseophagus': n[923:1511] + n[3074:3394] + n[3434:3832] + n[3942:4084],
        'z_linie': n[1511:3074] + n[3394:3434] + n[3832:3941],
        'magen': n[4084:4891],
        'pylorus': n[4147:4717] + n[4851:4891],
        'duodenum': n[4891:5334],
        'inversion': []
    },
    "Real-20200821-0010-0001_2": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': [],
        'z_linie': n[2975:3016],
        'magen': n[321:800] + n[963:1194] + n[1355:2975],
        'pylorus': n[321:800],
        'duodenum': n[0:321] + n[800:963] + n[1194:1355],
        'inversion': n[1751:2327]
    },
    "Real-20200821-0014-0001_1": {
        'ausserhalb': n[914:1479],
        'mund': [],
        'kehlkopf': n[372:896] + n[1520:1613],
        'oseophagus': n[1613:2681],
        'z_linie': n[2591:2681],
        'magen': n[2681:3706],
        'pylorus': n[3217:3706],
        'duodenum': n[3706:5335],
        'inversion': []
    },
    "Real-20201124-0001-0001_1": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': n[350:575],
        'oseophagus': n[575:1196] + n[3527:5340],
        'z_linie': [],
        'magen': n[1196:1839] + n[2751:3527],
        'pylorus': n[1766:1839],
        'duodenum': n[1839:2751],
        'inversion': n[3133:3295]
    },
    "Real-20201124-0001-0001_2": {
        'ausserhalb': [],
        'mund': [],
        'kehlkopf': [],
        'oseophagus': n[0:553],
        'z_linie': [],
        'magen': [],
        'pylorus': [],
        'duodenum': [],
        'inversion': []
    },
}

annot2 = {
    'ausserhalb': {
        "Real-20200821-0014-0001_1": [n[914:1479]],
        "Real-20200821-0004-0001_1": [n[2502:3607], n[3859:5075]]
    },
    'mund': {
        "Real-20200821-0004-0001_1": [n[2502:3693], n[5075:5132]],
        "Real-20200821-0003-0001_1": [n[1089:1173]]
    },
    'kehlkopf': {
        "Real-20201124-0001-0001_1": [n[350:575]],
        "Real-20200821-0014-0001_1": [n[372:896], n[1520:1613]],
        "Real-20200821-0010-0001_1": [n[738:923]],
        "Real-20200821-0008-0001_1": [n[1026:1220]],
        "Real-20200821-0004-0001_1": [n[2412:2502], n[3750:3859], n[5170:2535]],
        "Real-20200821-0003-0001_1": [n[1148:1282]]
    },
    'oseophagus': {
        "Real-20201124-0001-0001_2": [n[0:553]],
        "Real-20201124-0001-0001_1": [n[575:1196], n[3527:5340]],
        "Real-20200821-0014-0001_1": [n[1613:2681]],
        "Real-20200821-0010-0001_1": [n[923:1511], n[3074:3394], n[3434:3832], n[3942:4084]],
        "Real-20200821-0008-0001_1": [n[1220:1893]],
        "Real-20200821-0004-0001_2": [n[0:618]],
        "Real-20200821-0003-0001_4": [n[0:1029]],
        "Real-20200821-0003-0001_3": [n[2677:2744], n[5092:5335]],
        "Real-20200821-0003-0001_1": [n[1148:3027]],
        "Real-20200101-0002-0001_2": [n[4932:4992]]
    },
    'z_linie': {
        "Real-20200821-0014-0001_1": [n[2591:2681]],
        "Real-20200821-0010-0001_2": [n[2975:3016]],
        "Real-20200821-0010-0001_1": [n[1511:3074], n[3394:3434], n[3832:3941]],
        "Real-20200821-0008-0001_1": [n[1893:1987]],
        "Real-20200821-0004-0001_3": [n[5066:5335]],
        "Real-20200821-0004-0001_2": [n[618:1183]],
        "Real-20200821-0003-0001_3": [n[2677:5092]],
        "Real-20200821-0003-0001_1": [n[1676:3027], n[4354:4441]],
        "Real-20200101-0002-0001_2": [n[4012:4128], n[4354:4441]]
    },
    'magen': {
        "Real-20201124-0001-0001_1": [n[1196:1839], n[2751:3527]],
        "Real-20200821-0014-0001_1": [n[2681:3706]],
        "Real-20200821-0010-0001_2": [n[321:800], n[963:1194], n[1355:2975]],
        "Real-20200821-0010-0001_1": [n[4084:4891]],
        "Real-20200821-0008-0001_1": [n[2387:4652]],
        "Real-20200821-0004-0001_3": [n[0:5066]],
        "Real-20200821-0004-0001_2": [n[1183:3000], n[4890:5139]],
        "Real-20200821-0003-0001_3": [n[77:2677]],
        "Real-20200821-0003-0001_2": [n[3686:4652], n[5109:5332]],
        "Real-20200821-0003-0001_1": [n[3027:4929]],
        "Real-20200101-0002-0001_2": [n[4118:4929]]
    },
    'pylorus': {
        "Real-20201124-0001-0001_1": [n[1766:1839]],
        "Real-20200821-0014-0001_1": [n[3217:3706]],
        "Real-20200821-0010-0001_2": [n[321:800]],
        "Real-20200821-0010-0001_1": [n[4147:4717], n[4851:4891]],
        "Real-20200821-0008-0001_1": [n[3470:4652]],
        "Real-20200821-0004-0001_3": [n[511:770]],
        "Real-20200821-0004-0001_2": [n[2580:3000], n[4164:4300]],
        "Real-20200821-0003-0001_3": [n[821:1340]],
        "Real-20200821-0003-0001_2": [n[3986:4652], n[5075:5109]],
        "Real-20200821-0003-0001_1": [n[4788:5335]]
    },
    'duodenum': {
        "Real-20201124-0001-0001_1": [n[1839:2751]],
        "Real-20200821-0014-0001_1": [n[3706:5335]],
        "Real-20200821-0010-0001_2": [n[0:321], n[800:963], n[1194:1355]],
        "Real-20200821-0010-0001_1": [n[4891:5334]],
        "Real-20200821-0008-0001_1": [n[4686:5335]],
        "Real-20200821-0004-0001_2": [n[3000:4161]],
        "Real-20200821-0003-0001_2": [n[18:3686], n[4652:5075]]
    },
    'inversion': {
        "Real-20201124-0001-0001_1": [n[3133:3295]],
        "Real-20200821-0010-0001_2": [n[1751:2327]],
        "Real-20200821-0004-0001_3": [n[3330:3473], n[4160:4252]],
        "Real-20200821-0004-0001_2": [n[4300:4890], n[5068:5139]],
        "Real-20200821-0003-0001_3": [n[77:587]],
        "Real-20200821-0003-0001_2": [n[5216:5332]],
        "Real-20200101-0002-0001_2": [n[4360:4600]]
    }
}

"""
num_frames = {
    # sum is: 106223
    "Real-20200101-0002-0001_2": 5342,
    "Real-20200101-0002-0001_3": 5342,
    "Real-20200101-0002-0001_4": 1106,
    "Real-20200821-0003-0001_1": 5336,
    "Real-20200821-0003-0001_2": 5336,
    "Real-20200821-0003-0001_4": 1967,
    "Real-20200821-0004-0001_1": 5336,
    "Real-20200821-0004-0001_2": 5337,
    "Real-20200821-0004-0001_3": 5336,
    "Real-20200821-0004-0001_4": 985,
    "Real-20200821-0007-0001_1": 5335,
    "Real-20200821-0007-0001_2": 5336,
    "Real-20200821-0007-0001_3": 2997,
    "Real-20200821-0008-0001_1": 5336,
    "Real-20200821-0008-0001_2": 5337,
    "Real-20200821-0008-0001_3": 5335,
    "Real-20200821-0008-0001_4": 4480,
    "Real-20200821-0010-0001_1": 5335,
    "Real-20200821-0010-0001_2": 3810,
    "Real-20200821-0014-0001_1": 5336,
    "Real-20200821-0014-0001_3": 5337,
    "Real-20200821-0014-0001_4": 4724,
    "Real-20201124-0001-0001_1": 5341,
    "Real-20201124-0001-0001_2": 761
}"""
