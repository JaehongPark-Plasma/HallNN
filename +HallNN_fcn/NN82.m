function [y1] = NN82(x1)
%NN82 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:30.
% 
% [y1] = NN82(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;1.99865947322784;1.9990017050108;2;1.99839954032957;1.9992272778671;1.99975596150191;1.99836723130233;1.99982293109337;1.99824339295413;1.99841393245975;1.99982017616721;1.99943992221578];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.37088475734303638642;0.037799588814311342599;0.56170763343233587328;-0.64514184266785368216;-1.7622518485876588201;-0.56683477221471068042;-0.017916517889283809645;0.063631100433188286791;-0.12257709414980760831;0.03371456639253091192;0.32516265402332389378;0.087050808781639926748;-0.052534902526792714994;-0.12289769311326398382;-0.017397217406628202874;-0.64168357166450851192;0.34515020336266283296;-0.48812796906746891423;-0.22013475355706063619;-0.63015854708044127808];
IW1_1 = [0.37981936151317835515 -0.39002828744642575876 -0.42576200673455072421 -0.28934042719531238852 -0.27678102785441460743 -0.18582263545019284123 0.66994740812318609446 -0.64494846690518381482 0.33391789966681534318 -0.28543401656990569082 -0.059312338499790694113 -0.29241293708398019557 -0.061840124895972591046;-0.59990811895131634923 -0.72447793166643914553 0.29185871524692907908 -0.26564051264918414619 -0.042943659081618741624 0.3942931504761203465 -0.60295753867515300417 -0.41751670616273006686 -0.015783335485034528933 0.63642925898728586187 0.12394018389816159609 -0.08065764738704837844 -0.080778253219128251517;-0.25887891796920342591 0.15134472430222570272 -0.36214519475659823211 0.98021715234528550287 0.40315258684883881202 -0.54860915980606306253 0.1868117738492154345 -0.5089491473044202019 0.37348237522999028215 -0.096103594244029519333 0.13905109405515567178 0.34944746797365705548 0.15754061528486018595;-0.98415123798290882196 -0.22761140808553262382 -0.30293391600411634368 1.6072055765963193519 0.043357980789218319995 0.033484265903905158224 -0.051846804312436357842 -0.031176642322523033524 -0.047758826201707071935 -0.031174716623238410418 -0.22281054047383203054 -0.045877665648949808586 -0.026523061399111921149;-1.6987026027740017575 -0.71348309430587941726 1.3123340328910848562 0.045542640109016319094 0.25066130943233938533 0.42424621263543055294 -0.82000539426321461534 1.2676550809688849064 0.0011239072003150453113 0.55283253628328032914 -0.0025391593787210879445 0.046590482732408219912 -0.063899549878459416252;0.14924427570975887236 1.050638151180990576 0.041558337701700934219 0.10764073224138014095 -0.48683202328438102136 -0.061721138242974822596 -0.117376893407986016 -1.206194014054521535 -0.03109067193133703555 0.36928261193091121983 0.045303853152526475656 0.045170763633657448621 0.066109395661360162433;-0.78510217526480308159 -0.19890663440344502622 0.71859132041982509698 -0.083266647050426939591 0.73939629321491417979 0.60701045645212714508 -0.07873453552888645679 0.0024384412880998213025 0.31681544577749792824 0.44063484332046742864 -0.049826908911320501316 -0.14046171310194666138 0.20494946153806337108;0.56502851478392501772 0.027877257488379100087 -0.25509961224356014764 0.16125690543668669275 0.55208432313711497397 0.11407438091740258745 -0.1229215259261667581 -0.20518292644833222016 0.051135401606413041775 -0.083538480729738531783 0.083543197834782881306 0.15435477274438452744 0.11077216209753214249;0.30456497597311688041 0.72406677387986906336 -0.77440928956667498539 0.19538985295716296275 -0.46278415416171297947 -0.032624544723228968146 0.041927984557968887847 -0.19518266412496959594 -0.061659128496299682454 0.61238900422868580709 0.28864118684888134814 0.10193814231450508412 0.048579518958620983293;-0.77253006336488461159 -0.54324778024058406611 0.79658114154828651987 -0.3388244579679119961 0.36461651102546149827 0.31543909624246657675 -0.13996019129077857945 -1.4724663796722079478 0.058397842060278710996 -0.099923724897553786728 -0.25430914691489131529 0.0015132624716831001577 0.073247616662834191814;0.1141250089816130131 -0.2667614619182371305 0.033235963204423665807 0.22081328864173141446 0.092670284315912287987 0.27747805537693492894 -0.047444812418297253875 0.32633581819531165547 0.12673774357016134862 0.24525902004169555792 0.39205261498731086567 -0.0065535953432473816396 0.24511333712987662392;-0.22120846071022665091 0.55336789819335530094 0.37368656503882774489 0.17554899544102914311 -0.40852657022801275311 -0.61205364601893563758 0.26221176596443868112 0.55920090589183879537 -0.046917085858957201239 -0.61765393227843223833 0.15071169428161926285 0.11284967600793018028 -0.028161156793342058402;0.19877651428140405421 -0.046672486154030072558 -0.52391565164468134608 0.095797358291788051177 0.48007496924919301762 0.061005391823542906571 -0.55068799728198492005 -0.10745721028141040632 -0.075029498109955766738 -0.27465743071942944065 0.1530943587919005866 -0.097557002545935189719 -0.0011251299451980095907;0.27078937882193981901 0.255631028548746142 -0.14058989126555024307 -0.46566657300952907228 -0.098984899548882246401 -0.014396146089321265951 -0.075010759153205125438 -0.024343961483522309763 0.064912914746405023236 0.10028251531551568565 -0.30887131490006303203 0.071591896039049751632 0.08101817301759593748;-0.70861866665298334222 -0.15883593504323637036 0.17608923819618280127 0.30862844999554883696 -0.06114762021950075066 0.083331561685263635941 0.054842784379773037995 0.16866179183039903267 -0.0018953023498192611996 0.084669083967604474861 0.025472233101711025394 0.012145968155025254828 0.00065777555874754517658;0.66007719439787870819 -1.0666052899772924256 -0.79933049535259304008 -0.78564092204953739351 0.001345678411276994774 0.37570216344623064142 -0.15283938157749471509 -0.51213251428893524775 0.13265130541082831295 0.29097356465005025372 -0.52494011083996883205 -0.34228344159879958708 -0.1326965492711058292;-0.24077919023503313589 -0.024321454575202482162 0.33067950769762899377 -0.41723128479612570096 0.17032106099161703483 -0.15196419155480150875 -0.098951683690793323933 -0.41890262428155145003 -0.034410910035102000581 -0.095822665365627432421 0.067143399139198714498 0.021034139966615152811 -0.0182600542758818013;0.058684716867722849787 -1.0142377455239255379 -0.49317326540510836308 0.34074085362542189115 0.43594069624431452947 0.55894722995889301043 0.39093651576098825684 -0.12560876283803001918 -0.017659070788963320692 0.12197360539125369494 -0.20770910479601470411 0.016197315202772479686 -0.036415174470935715578;-0.62801903964455851881 0.1641406392340843956 0.25769408177615826805 -0.32463860748156303515 0.16605707953471623983 -0.19198353590938363467 0.21148767853275213247 -0.0090660167089083101843 -0.050274626487773380057 0.29178513887301177343 0.042026641850801980949 0.095528786390540937346 -0.10152270003512489649;0.24968080353603763766 -0.74673090355565285847 -0.23219359584954626929 -0.2270336221569550228 0.77290739913427175534 0.26974878257056772446 -1.2220808406830689918 0.3786379665174373077 -0.14149490484303364757 0.57100987700665628566 0.10367648677843842309 0.0089344541453613823129 0.15470548103235221005];

% Layer 2
b2 = [-0.08303445861454457344;0.089310390774989575902;-0.82601302347655991554;0.41283586272966038555;0.01317760303320690031;0.031834794647096667508;-0.28973007705059883321;-0.20813703847012526849];
LW2_1 = [0.030308336572600694248 0.074009461890680305896 0.085676096344340701982 0.067928351973225928329 -0.10473814756894145317 -0.010566886658841187496 -0.0832301309672422579 -0.029174919942383147758 -0.034834829935364255982 -0.10008358805280878623 -0.083805017246985011314 0.01926890807098027153 -0.34423508256121959636 -0.27528699313148496941 -0.2604089888858068802 0.056978636379523633915 0.19425847518896513511 0.12839903776940880809 -0.21579770703549835953 0.17475961893978744777;0.39661801722943335058 0.24539287507943269295 -0.12051575467927863605 0.10710589676014392602 -0.041961141794739843791 0.16094148241761227491 0.013332108763279101882 0.16109980454508285308 -0.14153054553575419861 0.4319055177656763167 0.24276642985788732254 -0.14133035612170621875 -0.15751589473241764394 -0.16154245635783354151 -0.092362084710190750503 0.067035365350598688505 0.17966090238281157143 0.069687871967723732936 0.0016953447369269045858 0.1174180606735673732;0.93928130773560136735 -1.1713762090618133893 -0.7931887701010029712 -1.2564687176647979516 1.6712852126399171393 0.92405731040152216949 0.56940542728896359481 -0.82642277667225205739 -1.1676461784379519315 1.8091950517298451828 0.032646823338315207608 0.39924720654358603067 -0.93239210242502090775 -0.33593192464399446173 0.5938120868797603924 -0.86546419214522596342 -0.011454870069216766676 -0.70572766581413381992 1.0107961978140975745 1.01896257846454108;-1.003322942068473056 0.3743169984849517018 0.77523106544212116997 -0.048983796937939083627 -1.3182151748583195161 0.015222641436371840001 -0.56224036306459945767 0.11130423889152281369 0.62474696018278774723 -0.44292706863963399178 0.016032233604975226066 -0.4305755697918016156 0.63079043098792852984 -0.26026633035927426718 0.71401311126958844078 0.66363825798741482576 -0.54482965878115452441 0.46277764775592711288 -0.60371035656487770371 -0.46337165371347355247;0.03226159882762677511 -0.32126731865031959989 0.0083942301313657645412 -0.056006535355629091799 -0.57528092308303080404 0.091793708325170475515 -0.1293728877779475317 -0.13193252692990006802 0.29142176816697140662 -0.10015937812369532511 -0.005040621075512442123 -0.13302754297947366746 -0.17470814650657245926 0.071235754036456933047 0.019793763418332090132 -0.13547615357712672335 -0.42487765532500221255 -0.13435445092376407938 -0.098625397025241437476 -0.15597643593066959355;-0.25374381467771767174 0.33937782390494097839 0.016270851304514617752 0.31835230189595625783 0.14054449812725389179 -0.060892876548222146793 0.058948854113764145857 0.33297817240590382726 -0.056707567029388804281 -0.30725867431893555404 -0.25809804903171068569 0.18388031178198982474 -0.021158939977206758093 -0.20716153893320540802 -0.64991642342534294841 0.10417577912375176885 -0.15227488931658722526 -0.030369314315898430362 -0.03189409904338566526 -0.056555665213956843318;0.31180023952202318149 -0.29929408560887776902 -0.13381887808827089281 0.03214841645662320363 -0.20177588639102764234 0.14816923436393003577 -0.11320569260287570434 -0.14248441579402906787 0.20755964629977829983 0.23025520777895455704 0.30227172514133088255 0.49954500434760523708 -0.041262271882220612207 0.19134371694188745394 -0.14868817828488051225 -0.079874822526350044982 -0.44482420012208129245 0.35855964199316836538 0.05419258520853065253 -0.3309315955439769974;0.31134528965548630142 0.23479785980903122855 -0.18259196979858269283 0.41643934763643852826 -0.45133018328104146377 -0.019592609812952136578 -0.092389600122594589537 0.31347080888890138661 0.30917871017699749547 -0.21939613355637341274 -0.098799088225022105303 0.73288700607415635524 0.074115866243069331021 0.37486194837455211193 0.65008535653380827846 0.064243033261088017261 0.035985944491345685237 0.21149756286025570007 -0.14900311132471505404 -0.71117193381414156583];

% Layer 3
b3 = [-0.54017486966470484866;-0.47722022593714952254];
LW3_2 = [0.24228306893047626969 0.39228885660802215174 -1.611197820632957578 -0.59988538402879754141 0.58026524966974746267 1.1145958999216660779 0.81138294417140433001 -0.53643311424375872409;0.55981223714595274021 0.27854856763034963052 -1.7513055081923618594 -0.77070723477059810413 0.46200746403486120384 0.83529369496150385199 0.67776238397491050947 -0.43281754341931633068];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [2;2];
y1_step1.xoffset = [0;0];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = tansig_apply(repmat(b2,1,Q) + LW2_1*a1);

% Layer 3
a3 = repmat(b3,1,Q) + LW3_2*a2;

% Output 1
y1 = mapminmax_reverse(a3,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end
