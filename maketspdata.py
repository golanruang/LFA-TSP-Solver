s = {'0': (0.24696783282943102, 0.8259250597628464), '1': (0.26506440374867923, 0.7181896836165801), '2': (0.2090818625135087, 0.4897986139726963), '3': (0.3452279716591782, 0.493329898512896), '4': (0.3870789482490813, 0.3612784887823054), '5': (0.5036200883927727, 0.6044966097702), '6': (0.7321405955079443, 0.1998896708614556), '7': (0.7458184158814926, 0.22252817079475462), '8': (0.9190421708179292, 0.9782575405421327), '9': (0.5903839067851631, 0.8251009655135882), '10': (0.9781223848984217, 0.7871445287819795), '11': (0.8566356274050025, 0.40727671035773205), '12': (0.10088596783264925, 0.08366346464771368), '13': (0.9015826995986499, 0.7389200911234375), '14': (0.620068052039078, 0.9728773204497991), '15': (0.21807501057077427, 0.13924281652538884), '16': (0.7181772322961892, 0.7521713425776815), '17': (0.599916317374627, 0.7328473696481721), '18': (0.18654945744574059, 0.6269392229720654), '19': (0.3101858761414471, 0.08783678010524554), '20': (0.3145813336557156, 0.059440144362356895), '21': (0.496973592446185, 0.2047400824582647), '22': (0.2106176362835236, 0.025024269993921933), '23': (0.5749078472797905, 0.31897320133098817), '24': (0.07479041745363679, 0.47226705240248845), '25': (0.024606204650597152, 0.8823312840110479), '26': (0.6712728378569647, 0.8123540490401927), '27': (0.6004597872207644, 0.6067917363867461), '28': (0.7853981288335318, 0.15271508757401053), '29': (0.3186659376299945, 0.46660665538633106), '30': (0.5514968157986697, 0.19223117986501592), '31': (0.48281266822780056, 0.691023059715553), '32': (0.07255816725530395, 0.4858338150685422), '33': (0.5567289469217915, 0.014558971781019947), '34': (0.1948900191056393, 0.8095105416201367), '35': (0.05125445595206268, 0.8204803653258271), '36': (0.2752113894775028, 0.7604664759767163), '37': (0.8724289549208718,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.7879568870732278), '38': (0.47434657376009315, 0.017209777541553173), '39': (0.02117127667574492, 0.5475261885166002), '40': (0.7619630492089788, 0.059754906225966575), '41': (0.016531911118901177, 0.7507082342397065), '42': (0.29131568650845807, 0.540690640791762), '43': (0.44859671314356464, 0.3871651450616401), '44': (0.7008891207004546, 0.6729698846805614), '45': (0.34599098373693404, 0.5086299550123676), '46': (0.07103711199465412, 0.8842098982615386), '47': (0.6713511654598802, 0.33916147372251015), '48': (0.15970517795152384, 0.956724109000205), '49': (0.7459152718421106, 0.6766524995325781), '50': (0.3654950770398706, 0.19268485753298104), '51': (0.29546868154061023, 0.45653512996583856), '52': (0.5763508216527148, 0.9322930198602003), '53': (0.6576429985378488, 0.33378464594161883), '54': (0.9318512870286081, 0.5155427472107772), '55': (0.03993050053743441, 0.4540192955019292), '56': (0.053511032352340004, 0.04348430064255793), '57': (0.7401689629507642, 0.9229455179499783), '58': (0.678724332303119, 0.7318411599166774), '59': (0.9074193830856522, 0.9234429744588104), '60': (0.24956281212749387, 0.516836800649812), '61': (0.21059971427060808, 0.33714732221937826), '62': (0.6239949487647807, 0.7668951517371905), '63': (0.1702888290260295, 0.48454488387894934), '64': (0.03727529693035547, 0.5310086939297386), '65': (0.5776992602255487, 0.8543960098780505), '66': (0.8240366615826514, 0.5444851243423051), '67': (0.18439483301529636, 0.6479524035925764), '68': (0.3084629219430224, 0.020178922707812963), '69': (0.5337908726197288, 0.6255108794697345), '70': (0.7534399358069017, 0.9780763284509355), '71': (0.8403523971274452, 0.7589751778576121), '72': (0.6623449620352081, 0.4816853406272167), '73': (0.22001711044238992, 0.40472999812345734), '74': (0.00011202820696165627, 0.2351846765385851)}
print(s.keys())
fname = "TSP%dcities.tsp.txt" % len(s.keys())
print(fname)
file=open(fname, "w")
for key in s.keys():
    file.write("%d %f %f\n" % (int(key)+1, s[key][0], s[key][1]))

file.close()
