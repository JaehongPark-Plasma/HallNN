function [y1] = NN27(x1)
%NN27 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:27.
% 
% [y1] = NN27(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [2;2;2;2;1.99869567372725;1.99905098464885;2;1.9984153427732;1.99964917429969;1.99851988741562;1.99818700708193;1.99909810321994;1.99996040628946];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.17923675067470901112;-1.586656178086064406;-0.53623506665788001069;-0.18092485321620493943;-0.17822257916361028407;-0.28882599612403253575;0.19346698828686328953;0.25134281103367306365;0.029083851214412613506;-0.1044809319940786102;-1.1643924907876548325;-0.75511875181155541092;-0.35934123363326225808;0.39342249406539847589;-0.14142045575807526969;-0.40431570182567300176;-0.022851898101586397977;0.026225782597934350904;0.098048471248757734786;0.46576572692705975332];
IW1_1 = [-0.10826911294947261233 0.56742656226214061022 0.38049357734025784916 0.094707753432237695712 -0.28690727672396670878 -0.47207173005256963805 0.54277191266994917473 0.16610192130622025708 0.01407969422862944027 -0.5475520784695290244 0.18440893235251662463 0.097837686766938408711 -0.032657997250266634914;-1.249080141385532805 -0.92276229710023771613 1.6252181791697986046 -0.89482191818358514102 0.48924405218567662557 0.37873054279801704469 -0.8543314781333344321 0.91351134577098547229 -0.068660079769103141145 0.60419841208203362015 -0.24879868807025920519 -0.036741307116560033286 0.044836969425728026573;0.4343866923561879867 0.46404543336174419732 0.025354237693060898556 -0.12331874794451591626 -0.15910124692228139187 -0.28103661154130826771 0.12716187629982866891 0.74973635570087360236 -0.081264333590036610944 -0.003736448157893926085 -0.033200075167783421604 0.031977580903610261509 -0.029826718333095685043;0.69759658278911562679 0.33039173966029145024 -0.31649995558019289321 0.21257587260304450227 0.019802607978652166099 -0.21354773416418501575 0.14775310554279233455 -0.32370871776370230277 0.054377281663402950163 -0.41596424473790627463 -0.32289662018993131598 -0.0015167870222072615469 -0.022824589116624265095;0.38494584892128258513 0.016153319123467718005 -0.45548267724438890047 -0.13699219093023692895 -0.22220241644543825554 -0.22863305033643027686 0.38251222755814801024 -0.43791872379576540775 0.2021473691840112441 -0.33538640475088432602 -0.40959364594378194768 -0.57012272500148630794 0.052420311297921531879;0.48045037495820536178 0.26993554798350738455 0.88334064806011458604 -1.5280462578451785571 -0.13291605218088320406 -0.099050755363746120774 0.20564931999219765446 -0.072717595216508709721 0.0069632333512222692462 -0.11910813744323213437 0.16191260975938734368 0.037986745733337469455 -0.0015126375994150920504;-0.44221873130190958445 -0.016491290542855630702 0.19908503338795344173 -0.098493053693736773435 0.19619898798058044154 -0.044258152028090462504 0.024643964046028751097 0.079466585053222946189 -0.075724196602169474568 -0.41955364009778339796 -0.21395249179300926401 -0.0043063537296978192728 -0.084584650509021530596;0.57784188032332683971 0.16519778990945183095 -0.18648293356427980694 -0.19260640225573028017 -0.13426861240338142101 0.11627739808740132632 -0.28077522411320371409 0.82105899262441073283 -0.25228652885143326712 0.75693097078332460814 0.62596240997043051113 0.3300945399829594562 -0.073818859601203282161;0.15182253732255235334 0.089088657104940668918 -0.19308345771238599742 0.36430754030372136665 0.62023621772058623591 0.040000174671139629434 -0.72787950376536381114 0.064289238263433676202 0.029524434989114587552 0.035289368906150753169 -0.047223395041726422272 0.044711242315189032859 0.0091955064718610354801;0.080400811781560652114 0.011381889528642281928 0.18406938861097621452 -0.3837414730276410002 0.16258701866991262519 0.036264380958173206282 -0.30381050338726200977 0.1610190727834881208 0.15379776062647068646 0.067161883158630780732 0.19644510883984897531 -0.066244531745527329702 0.0059638678601664088866;-1.1946749155133040698 -0.88616145853617422912 1.4568984705245111577 -1.0475033684659467159 0.49058791251800337729 0.50517375848693768514 -0.9604881023940774476 0.37153493372225110392 -0.059858436290653696532 0.77768377342193772961 -0.10775602805055504996 -0.10097278018946349853 0.054874836964848544563;0.63697576167114655643 -0.33188483924412343073 -0.6610364589385508971 -0.82089317395064764771 0.039327054139584372017 -0.22825140453409492602 -0.36040898142255101133 0.12497272362446776306 -0.25949077270318909871 0.19731598686724327596 -0.45788742127360143597 -0.35914282520107077001 0.087577340981758108884;-0.27154697609449918172 -0.34337950515415399666 0.27565406222550076221 -0.086310203152695649798 0.37012526992207295251 0.85476231618843512283 0.24143872112657430162 0.38697306929971497791 0.030009271319138355189 0.41823577003255496543 0.17232029197725856617 -0.36945298667260123571 0.014533590128914392059;-0.34778295668388115036 -0.15283503216382965384 0.3357957853236804846 0.049633713159453703156 0.069284415021902620579 0.2242377499321218659 -0.32421897902127827518 -0.49805370873514659413 0.046290988697790233508 0.095886021215676575657 0.29114046586390429638 0.014375462902218220582 -0.003847436512146056118;-0.29571772063496215566 0.30354256062985973674 0.78546304683211465303 -0.70711280064793413658 -0.14185641842372037624 -0.060519749980352047225 0.11177099916291260984 -0.45426361812014381947 0.023034443066067292194 -0.15062809615884459324 0.47346185083575892394 -0.025432188966539492209 -0.007784401050475713707;-0.019691484242220030909 -0.97638808980193969589 -0.66133132251065662732 0.1522068693056943256 0.53825369952770940607 0.67479270287417547625 -0.63032147322092335617 -0.60464037504240109033 0.091943280184281384049 0.35477088045963867113 -0.10381756072520691159 -0.11611064033884210822 0.02800268881537248955;-0.12372014867997498111 -0.062950155428086532194 0.48285294124050676112 -0.63234569045232480455 0.21077989270114522169 -0.0080423816305925151932 -0.29981338723824013659 -0.091962268666336485778 -0.039410506547686671719 0.097385019670697639538 0.035197398325741410807 0.12662056671772109473 -0.0075997063983098802134;-0.033093061801043134607 0.017938295150369203135 0.13001267202863456274 -0.25305461136174517423 0.2543352637451277154 0.37555419633811876201 -0.231632397428592679 -0.095593099320734203905 0.022388333492534263219 0.24894836461030345198 -0.12240106035806416762 -0.052139378722268515942 0.057558069287696383975;-0.10289633965684273631 -0.3278443648694170931 -0.08605571177675241934 0.1196218208810839595 0.2546362224242965655 0.075095018357397513853 -0.011756355678880291843 -0.5554870732791042931 0.081026647837725340739 -0.30848982484860681952 -0.3816056216978213067 -0.065026018947216837618 0.046660979012508332908;0.015783674157728466714 -0.44089425019147943097 -0.1756820601410455096 0.017116289682764143643 0.32153329263526070347 0.20871426737605852653 -0.49729200376343180023 0.53412461414873690924 -0.015786303939636699839 0.26138186505572630347 -0.16936927686755684497 0.08688307955182522635 0.042040292123749219144];

% Layer 2
b2 = [-0.00075706884680174814373;-0.049402748593470484362;0.03236679870030341466;-0.31605034062516157434;-0.13281307594900174096;-0.2933043603603475713;0.081127457296731389502;0.20482231632820951939];
LW2_1 = [0.41468460541481860782 -0.58565646604956766996 0.10798520217122090092 -0.0077470108573391271295 -0.086861111461978973702 0.24453216435829774889 -0.39282678058155717782 0.32366640676792479692 0.37221661641046560609 -0.63928761280046397797 0.13381057888320638249 -0.072945586073015308837 0.077404283139421201088 -0.11244009999309582948 0.21941999621069457893 -0.16619437205293521864 -0.49038744501822439936 -0.56367306971381492264 0.012859539334346085232 -0.19577009749746018819;-0.13678456415956438685 0.078283701233951508147 0.011956700858638549811 -0.058245766846577742837 -0.076272509267389099197 -0.11880337410315257307 -0.018958851276012461673 0.09647605882741745742 0.0068342956087156401732 0.106505407489688661 0.18479398372758926161 0.021696000040202695347 0.1309117680383209692 -0.017141295711099464871 -0.013321905658389757848 0.073589413871285480617 0.20312275397979295821 -0.012810241217274103212 0.091057526745211586627 0.10369205913286735909;0.21832065523512902971 0.1667460158370797918 0.17521485048323356959 0.21065796554393698137 0.12669558235112893252 0.19061301946507769323 -0.10217245730402484127 0.13709441831999968908 0.43231404215402663427 -0.068157882351958409828 -0.21327058471679755103 0.262675117461824148 -0.06510590593928382297 0.65393706753215374849 -0.31662787882052995103 0.3263372832278382818 -0.094681738889286853889 -0.11688832895694818703 -0.17248594163161487525 -0.52929696667655190545;-0.39879582083463821496 -0.45853267566939198696 0.14286877045939766484 -0.19426709805403072351 -0.090583062214293186276 0.5466734519758652544 -0.13315560796211800554 -0.13465704463250205425 0.073250765952590488062 -0.3047705515148703137 0.30595549500565860646 -0.24221440022374221557 -0.010324502864809088215 0.24120513622398398468 -0.46343050759949916406 -0.19928055499078195512 -0.51210785308196959953 -0.48950421147019518298 0.44117101740011688271 0.030522829924645923649;0.68353279579591530446 1.0219496713204181404 -0.035503596867484492339 -0.27451421803628067719 0.50783737469592904468 0.31995853698482712435 0.50434935910729861153 -0.21978432889794588534 -0.17004469825711984732 0.2007064017245516474 0.73559278179632225925 -0.2653692794342461525 0.29433470425052887798 -0.0055566412906338123398 -0.057803476092328109903 -0.53216829149554534251 -0.078601891018016720181 -0.23094548623529514986 -0.048941283126505162604 0.12917943876932799774;0.1942902276556620067 -0.037845940724853768811 -0.071817533819910517323 0.069887751325255265122 0.30425673139131714784 -0.23836536196630514484 0.033750050779362647524 0.20446221219219445908 -0.39863150792300999781 0.013607414210495894988 -0.24497068681489969633 -0.068087432123731070344 -0.099382257273525884123 -0.083857254159200339538 0.21197189354976955777 -0.24775602876497751503 0.31067924553021858403 -0.26644494046128486264 0.22805412990529458361 0.10376537301014512882;0.81405415507813383424 -1.7959746260743165713 -0.97557577866553946144 -0.68123864062961991817 -0.52180368513756825166 1.1029830691090212191 -0.18897400400711286683 -0.14668076826485101916 0.68304991662102976235 -0.35813268647409168821 0.71814442254195054449 0.49579321476794019352 -0.33555497698939251716 0.2864502225227737453 -1.732630523687809454 0.80897410844804795715 -1.0813851881846212599 0.16953682193938018896 -1.5202394958762048827 -0.75004065269029307483;0.15806955873870137674 -0.49122733731410417324 -0.070653449513050622932 0.057795065318483870853 -0.3679481845181537536 -0.401695141112287879 0.081483390431713195734 0.37960307990885705509 -0.27913713394396172385 0.35044564846629189248 0.27407754075616919121 0.23499328015070050646 -0.18606366013292621742 0.17158613004522246803 -0.082182665031873394779 0.014514212340003957574 0.0029654674020251009861 0.016332208868887761188 -0.47210997149280103935 -0.28696220192100374557];

% Layer 3
b3 = [-0.16353091582997528186;0.04506637759045329511];
LW3_2 = [0.44386773330409279303 -0.35009179047091082415 0.69275274385418872392 0.82321967617972457987 0.6244174951356116221 -0.69739121023859362847 1.7952073096132703522 -0.80878727626969770803;0.41373310862359774509 0.17009045093045307984 0.78758340813067184705 0.90472668481694207276 0.66217056581469391219 -0.28458027851363992822 1.8248823131337208459 -0.82767935753317189995];

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