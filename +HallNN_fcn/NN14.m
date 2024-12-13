function [y1] = NN14(x1)
%NN14 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:27.
% 
% [y1] = NN14(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99928114710042;2;1.99990125791448;1.99969801769384;1.99826095060583;1.99804780651122;1.99951093009809;1.99867302982444;1.99842322075392;1.99949153808196;1.99834720425197;1.99947669008059;1.99818425212658];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.77094744018275518371;0.49388173605016211232;2.3241902076800897703;0.37971267308703249288;-0.52439422726811579789;-0.093319112855482527769;-0.48564372408673089909;0.40566556403955206234;-0.18677194143830699868;-0.020867415118757701858;0.72827778423227074178;-1.0238448980185395065;-0.089188759497906527907;1.2112885387812899474;0.090208540277158383303;1.3860913932079075295;-2.4742043457862874511;-0.31293233663459302241;0.46080534094501518494;0.15238186361040120809];
IW1_1 = [0.71765887168046293976 -0.81132795662491996591 -0.94001424424700197324 0.74507603184455706291 1.474802000881821229 0.40093977594585494817 -1.9096106983682770153 0.5642105714801544103 -0.053758145003621193858 0.6691601312111133959 -0.23798597842068031083 0.086300489779480335506 0.13272962090359910481;-0.68087759594094099391 0.84279654212455124007 0.37586063032050287536 0.06571088333509361723 -1.4522293871833225065 -0.61913502855835333971 2.0648680970478765495 0.11217297733700982987 0.063577226634819614115 -0.77817607079655748414 -0.18220770919637888086 -0.093434570986568182827 -0.14472647969735427664;0.96418741262279139548 -0.54786033024860747442 -0.70871833355981506131 -0.14712829723211978683 1.0671750109747613067 -0.25837615966706384274 0.27633438355321493507 -0.18914177118548800016 0.1725700113732118457 -1.6555863026346933076 0.60785438744673059297 -0.11938945350163887582 0.065507034165363700606;-1.8455247748104381955 0.13197689803966086708 1.5260821900822474362 0.3886598197392219789 0.21412196241431349919 -0.039093387489770731191 -0.29408304223725917348 -0.060456370471164797009 0.081434412524549346091 -0.17428390649246031097 -0.54284585996106171635 0.084922232134882674748 -0.12677414986088297866;0.47130368570065273026 -0.014744281027194969941 -1.3310289601523415115 1.5817805213456523905 0.18735301107658774988 0.072809457186832543529 -0.044617739145299012415 -0.29692691102062018782 0.084186531468535982792 -0.13365952651869106305 0.38781569925305425794 0.095373949241054342685 -0.021550256364323519542;-0.6142960475842318413 -0.44987788992985600078 0.42541240006650088423 -0.17692580673010649406 0.68300720107615708887 0.11036534895525394739 -0.67375895191823775843 0.40705949902949339902 0.020274057849660385294 0.05633188479990582781 0.16785190739827629547 -0.081575147391536387276 -0.00093443164272556948798;0.37755054822548705928 0.085443432550681902571 -0.59950682995736348513 0.51827094443572840898 -0.56123593639835722691 -0.31564950394132068601 0.18236236749909395805 -0.2204133206702733605 -0.66564336545483493435 -0.41934699006584330583 -0.15914018158921844348 -0.26771786720540430293 -0.33815090493215416156;1.2019860976420433385 -0.46717356673014820423 -0.48740283985889626406 -1.6093682971464762232 0.47336110534479824974 0.34103606511448653071 0.0046491259277721547205 0.12917551513664085605 -0.042212076253708111229 0.30087055124845679677 -1.0538522252241984134 0.020217001763807744552 0.022422994073633829848;-1.6920067209532099195 -0.79835880319034246977 1.719010494424359603 -0.76008344075374922877 0.15374199487026324329 0.49577711325046608382 -0.17977315092713991307 -0.31749669964013277346 0.080536610049576573545 0.061214934719930058638 -1.6574259705358018735 -0.087663573825171633991 0.04006047404485708896;-0.55156343408250652072 0.20793924552325396271 0.90653273738268735915 0.47264775263227137847 -0.62474274178633715238 -0.1446807150920222107 0.7445327007143643927 1.3147550765499420589 -0.079260368961789659514 0.60555987021076040921 -0.99301118647126584982 0.083943806521870653614 -0.036724311889457662839;-1.9688615252039618486 0.44173444290060465134 0.61244320558024700762 0.80734723494978144487 0.23132653739743308519 -0.38683601892249358523 0.78185222433878021064 -0.49043338824661514286 -0.29583579248989161359 -0.010880477901099278037 0.37743449176835042191 0.015499808520502811326 0.033013569663278716226;-1.4779730942964866003 0.44531939405366344209 0.58708755666252321337 0.39388856701907520863 -0.74466429517152132878 -1.2917995216354700716 -0.4505002052190610895 -0.13541280613337974836 -0.21886761899203183157 0.069569267983214347351 0.13830722279845164557 0.11680575854869439112 0.13228003808421906484;0.36880915541204489783 -0.69889714680212977616 -0.3630440954863345393 0.27461711071282191021 0.2173757990911373561 0.024271982163662126492 -0.028180124302898110211 -0.0070608355426047715586 0.064032176404690291371 -0.14616654749444171757 -0.073341064847764911305 -0.041124514011970515759 -0.0020571148632669180971;-0.020948201213819510336 -0.86017131262411028469 0.30167592117169911292 -0.17338568834860784706 1.008470195016712756 0.60879627727644436419 -1.5944861890258110737 -2.4543731793993646129 -0.002916642025911075041 0.73078539926533370163 1.1302549448408609756 -0.036193672192556244904 0.081364063896747551574;-0.17075021048427896364 1.2309989482303957953 1.1803474085135357008 -0.26399607671267794062 -0.91733615132129597836 -1.2283274308261367125 -0.26496443303666472602 0.52092268724431467497 -0.085447933498639425109 -0.56397703415105404634 0.50909133308910525795 -0.019547075183822034544 -0.10425212957452439921;0.54597884594684809034 0.25076812385502911251 0.13635715023897421183 0.48790164795437612755 -0.45232134350330521766 -0.17130466576412392565 0.90389455858151113166 -0.99661562684832427639 0.17265919829302933675 -0.19224891392754783759 0.38957281533577553523 0.23101999458258898335 0.067628321529462506745;0.067674101696373750725 -1.0127906797309591536 1.1818202791464971213 -1.7397410251563518724 0.29111391992971058507 0.25926039769026304471 -0.22989054633294825347 0.99764444550829722846 -0.046635160201587602669 0.39802338109543067279 1.403306460237759179 -0.09959669358866869715 0.14218186892780360076;-0.19291819402100537917 1.2215666616019171808 -0.086360225461831086058 -0.26495380200194834641 -0.05605069229517944257 0.10153877261581172042 0.079849880947095580597 -1.2067912038644299511 -0.18119722930272247341 -0.51605993914054881788 0.47243566349585941477 0.17233913454292376644 0.031083023799656189778;-0.038306805898168101721 -0.049641194371323367485 -0.59888093003380793444 0.443833999423084824 -0.1843245648290187666 0.0087162257740911464055 0.25533167895907404876 -0.23908876823123148547 0.027647367097022111648 -0.043589381038043023553 0.045833470131802334091 0.063559792899021458346 0.017265971491612689542;0.7315125419900442294 0.82087910640366112958 -1.3060994040002242844 0.78783288382283978013 -0.13102205776854805586 -0.57632529145730015951 0.23815872254533973496 1.0831498269886281616 -0.17627484118392258705 0.033809082742615070849 1.0130197709542341222 -0.055956737869974161603 -0.092457667890239611519];

% Layer 2
b2 = [-0.2854460655835744376;0.80883611798297738726;-0.56936571913967159997;0.09642021514648851932;-0.17228049247599658278;0.098455812216535418258;0.68193074018311683471;-0.16438847191632766087];
LW2_1 = [-0.33428537483997428081 -0.42082484082874277442 -0.24834249819728312914 0.0020184850793933764058 -0.13027625907344392298 0.87489535474383750557 -0.015567439710289105934 -0.25773707451574784377 -0.91334880787950489811 0.077296890155970576775 0.08077957057241320904 -0.078898044460847929771 -0.24323461769345172678 -0.076287795256524071341 -0.053500320865041722795 -0.11475464207871033384 -0.039639078197351677868 0.038284174071578157272 0.61739000023063417277 -0.17753392023967418112;0.0055236784771449179421 -0.029458311617066586241 0.17365785343423029152 0.035323101005204270153 0.27972653943218006578 -0.47930712212126258542 0.080137726102883097279 0.017566474593561465928 -0.29325469027490486118 0.061770678627874262001 -0.061505071514428424906 0.057160248541224714658 0.12974870968438317886 -0.068451444831195779051 -0.06727426714091704063 -0.025709938320177087856 0.083816976965720380033 -0.0312985318118128783 -0.5814584143939666161 0.029663077260279079039;0.47966961079818215552 -0.22073773344553859754 -0.63524366336299409319 0.21744086785406946394 0.31052464957988779037 0.025364036784530981372 -0.29496369581601300514 -1.3517262222991361931 1.5376485534738582039 0.52666245196677785589 0.45558663388453207732 -0.011818219306749297437 0.486567274622726198 -0.46350355086812050631 -0.21456525154670005873 -0.90495394786079919047 0.79266035731530837261 -0.49675176191193448627 0.64196069071772288162 0.019077293145401203606;0.17035953284208418812 0.23366269193049274411 -0.012149293431540416494 0.66043810101777855603 0.58617142239364339318 0.43767242714341003662 0.21544238924264236257 0.17669261735025557081 -0.079245448798574619209 -0.35172175934519583906 -0.096477517866517564782 0.040204777422053653013 0.051755229157482283975 0.13689076562527788261 -0.097735823611438316094 0.097680462817738095138 0.47202486885249361759 -0.10188410304729816902 -0.34954446387156168363 -0.057004664963259524257;0.21347431858824492124 0.34326389254373590454 0.47441099428911270852 -0.31194322628763021532 0.30246905709827626163 -0.58491130829527460122 0.02517772781006618471 0.022907808498648575546 -0.22102280152844958061 0.17788032301802025881 -0.1674510686399101056 0.066554611121760864911 -0.53050559380287576339 -0.035822938054202294045 -0.10413556208011855164 -0.11502561849347714418 -0.085371956152036743148 0.068808844721081591223 -0.072986180794283789242 -0.0069920015293179101079;0.67864120698188878933 1.5411831206557620177 0.88065220309130454179 -0.8906234749164284592 0.57463597841154545609 -0.87319348476159919237 0.30621903314149040432 0.79642685902329179992 -0.14442151509664855058 -0.55522669850416528359 -0.36608759862952733322 -0.14721942211339134055 -0.73340890501626865294 1.1693559687055032903 0.29159782638350861816 0.5331672471131665425 -1.1775000481768824745 -0.29606414467843789051 0.10421280717641125124 0.62000975040227201962;-0.26367680017749917099 -0.41277387042683877372 -0.54362216583000755321 -0.10809984510964817939 0.32035785423525087179 0.75161706692519159034 -0.12506405016306279543 -0.17864315807228114963 0.13857055614603619453 -0.053110504537792285751 0.13002512559044374085 -0.12639917801310329715 -0.15575902992017284787 -0.060825900909928232174 0.030264715087338300847 0.10951147813899445027 0.018017913258744279459 0.12231003315861191383 -0.70557265840580718486 0.079799293998633780145;-0.75539859204479375965 -1.7310316089177604226 -0.57118066134942635514 1.8629413906721501437 -2.1069571049982251942 -0.31226957970794899655 -0.85122326494683320242 -0.9477198551910507307 1.5517496019363548054 0.72878244268935488126 1.0789403450874410773 -0.83391837146402625347 1.047032238299751894 -1.7221110185015964245 0.93642827462089284207 -1.1937203666200433272 2.7287882412549033795 0.271869662296872594 -0.5598114263200152374 -2.3192967450940571439];

% Layer 3
b3 = [-0.42431134095869388334;-0.34250537500464567353];
LW3_2 = [-0.91496106824959433368 1.3809645880977603483 0.70277506681959489576 -0.21564820249168092925 -0.073437193491086460995 0.51317533308997387298 0.62740493464148283653 -0.41931531336789673592;-0.87553311544899514729 1.4017285145782882871 0.64537639198939200647 -0.2647945901879646402 -0.27597711518692297217 0.41294289112010412746 0.3838454459437395383 -0.48603782473797935149];

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