function [y1] = NN92(x1)
%NN92 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:30.
% 
% [y1] = NN92(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99813639976246;2;1.99860999189904;1.99823237196584;1.99816731020796;1.99889996870603;1.99958596611023;1.99867878106616;1.99810813541483;1.99882193510477;1.9982246085286;1.99945621862575;1.99903186273953];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.06002225646112362839;-0.38593157979818337733;0.095478710537820282722;-0.055166648635047228766;-0.082772707878739540388;-0.14100124684949097431;0.091147914031053947626;-0.25666559119946980028;-0.16043901603606056105;0.081286614939202347929;-0.21231529030535320479;-0.7222680938618351032;0.0021677400874930575353;-0.13804634897120676706;-0.85167344359498742268;-0.89765271297563675734;-0.17862259566621127571;0.3523452637760884909;-0.15705596118000048067;0.31398576896058533192];
IW1_1 = [0.099997539139090652061 -0.13805497864462823454 0.14591686489187349896 -0.08371057805582761413 0.2673172807010515406 -0.043304464475352431874 0.17887431646619603165 -0.12448186058973739754 0.10056703766075626716 -0.5001238959897994496 -0.16231204582263486502 -0.0236301674415033032 0.019125802091403201755;0.26173890321153364669 0.80672852688077356653 -0.11978845864548012812 0.16902731377696403325 -0.62049665036552015263 -0.69187614754013992968 1.2676967779871572972 0.93941697274076152446 0.053490822126385707824 -1.2260127968146714572 -0.082208813225170471184 -0.08985596726977308768 -0.032395282896837081232;0.10692088646394980189 -0.016365890603083594473 -0.20262421627548046765 0.4580772167513942339 -0.099842034547494643015 -0.082471428069373242664 0.10546199752700879948 -0.027220987100437698397 0.0051561333320166592015 -0.092076724183489180864 -0.041289894843659476209 -0.045695045198526747232 0.00029865691554392193377;1.6755216392676288706 0.13601109446944503745 -0.075096113080628357217 -1.9623023158524277587 0.039028538881628840884 -0.010163437074334612242 0.050455470377030901818 -0.16737794563308222573 0.022415025113109682958 -0.12032991376062868172 -0.20399740605301980301 -0.015997801805066897557 0.039364009661592841427;0.41705401091505300837 -0.058577082784746006827 -0.34592073966188024992 0.48918553970743139381 0.39276746887129798669 0.1577908668876097098 -0.022135080257415742921 -0.12940087757551055336 0.0053824684730859814752 0.0085555076099829222014 0.14339952520598736596 -0.00042059430726049291949 0.057479512483577514803;0.24390878865147727628 0.29121845308104593597 0.3672460777453941394 -0.54405715537157195438 -0.023315455977632074808 -0.063086426091212816081 0.052742176757165561141 -0.21769512189103135125 0.017886219118270419576 -0.11444926151696921401 0.076823383999163885605 0.082280985468867626209 -0.026329520658998406546;0.020065823326357059425 0.021734398741750541456 0.39756385720167819464 0.16263601356839621492 0.77828960413097902915 0.19004073136299459712 -1.0151178151260453753 0.3694553793062017033 -0.014806053916838632367 0.15195840226747017732 0.73560711521072308372 0.29089939605499431918 0.007152888740537285614;-0.58481416876089287182 0.52928459173269848392 0.89383795999865112147 -0.42420570837560950261 0.03991452830726210238 0.14672832923225190283 0.33638169182795790713 0.54953110730809651141 0.020669637014642347483 0.32823745849339408531 0.13944612896961933179 0.15005047890773257069 -0.12053884754095219034;-0.20129774944014791616 -0.142096502906017913 -0.46603371420245059964 0.20143926757445695941 -0.53726056043413161767 -0.81223344746559267282 0.19968076632916975655 -0.31178784339864901165 -0.90620459551959886557 0.037880854798388395588 -0.47036340282897459719 -0.071288516117638978109 -0.24920646043690006244;-0.078520510151939212906 -0.9036222523638377524 1.0387228317329055827 -0.95040101866170934031 0.41240844524867720233 -0.20049887331216961761 0.082470759610858673794 0.311956521550355248 0.032562572752173661594 -0.42247080835979905489 -0.46403512799819085499 -0.023107863610131861021 0.033046848936325895485;-0.11570632921269879967 -0.65094214772093639265 -0.78351396671616213574 0.32152111786792053616 0.17439070823827082513 0.60491535565046639089 -0.065244372950888460116 -0.43655094099802654872 0.19661245096337848226 0.1083289346007154097 -0.37050689909092870966 -0.069771596976420963143 -0.088343675615370015342;-0.075694414989705915975 0.18026696781745385478 0.13400642796043613081 -0.79640484708471293018 -0.2023460210792948355 0.40168401252160740311 0.092083085067148642833 0.28516679615753826038 0.14250279198814161385 0.11864236822252896553 -0.054541203882526465008 0.13485663036892772682 -0.045737132611421600636;0.51565464276974204427 -0.29425109150797601476 -0.76102595708816411779 0.25484298945460936991 0.2043890999317933832 0.15224236660441159419 -0.20171206456347126612 0.4917497122638609941 -0.035778400012908800898 0.1901208975499317777 -0.13998437055283041719 -0.068221970646672694394 0.058663628878984686033;-0.14606382485572891317 0.0090187150677243019259 -0.0035895238166896215916 0.12055650180391958015 0.16096447596171922778 0.1623217332066709262 0.01222893126928816486 0.14950765439963811798 0.1256806492853643209 -0.1856814631206955557 -0.70348022632293016887 -0.025529147931086453027 0.0014052332802548242116;0.29458250211083253367 -1.0577139131740656008 -0.084395734439021014017 0.14049328727070281575 0.32926861657446565568 0.86955395447036454293 -0.96180462699623303635 0.63166031737251604472 0.015171936154874178168 0.84438268749936840063 -0.12168803521420955283 -0.10303590631371738195 0.011218444592962107526;-0.31119130173909126391 -0.76316097569604035566 0.57747288038019217726 -0.68333935414078339488 0.8858591363457780643 0.28681117851026788346 -1.2277528854819457216 0.48320116491211834164 -0.13085064445356400786 0.82351389150880649481 0.11364744007825049177 0.042306538281159412895 0.13244893300250429324;-0.045056552964381782478 -0.55974712408093918459 0.451302265687509363 -0.21314705153927165915 -0.076534388744114112413 0.44633414011345212691 0.028875579721541439621 -1.5413373657840783171 0.13171611250736436904 -0.1281307007290819755 0.14690284179403742537 -0.10424670109115641292 0.11025458118253268769;0.11934934999262628019 0.41422786635015806933 0.19586770306243558859 -0.1172502656135458593 -0.25346132807089616357 -0.34373413508161881902 -0.063199092499237982667 0.044653430530264270892 -0.052193602260540587767 -0.043165397119578112484 0.025707117736056384061 -0.011142623528467322791 0.038890170606713563017;-0.44947688994409712437 -0.23924276282758694112 -0.47262900276381147879 -0.057944727639857633583 0.4806712175971674661 0.15474496180448785165 -0.19254798981016035819 -0.89833217909633755749 0.055285609619456867192 -0.43598878847932037717 -0.56542208480209399379 0.04343676693119151444 0.052870843223014156453;0.76063579331575914555 -0.045943225251805171194 -0.68440664256307226498 0.84829176962028840503 0.37947580838721817198 0.0042802234923579153894 -0.084661533566984220611 -0.2120173672877302129 0.1806057530804003608 -0.3072064439967770455 0.27580110460364670733 0.38160607913790284274 0.14598877745625815638];

% Layer 2
b2 = [0.16154343345804550447;0.2378478654693795824;-0.7694835953581696808;0.57539743655123187782;-0.048399377487780656515;0.01992391079027660758;-0.057302992946821507125;0.031473990538386877047];
LW2_1 = [0.0706506468156554851 -0.43939650957750697735 -0.25807963475772788264 0.09953275837010679139 -0.61756915613816898425 0.19059001976015471636 -0.13030404909096029997 0.090674821723915402516 0.11362448799654946419 -0.57530024217535402631 0.24511485222824405672 -0.54448817905788327032 -0.077146964823004968159 -0.31225149514701872056 0.11027259326005930951 0.11749915007442877923 -0.15699406418674025088 0.016478824685542431405 -0.12171748243850628868 0.19282010850672676172;0.014373072400612311311 -0.56758064913452122013 0.064659852907980097658 0.32642325591499821158 0.23856453491450019255 0.15551872447602438365 0.48208686180722959413 0.064458171897091839919 0.54300982068723502305 -0.11107929139122958706 0.32319995632092446414 0.053523907252129819534 0.45049805458061359387 -0.053781970820567466529 0.098608109713278602193 0.45376309602496001094 0.19146814092216246639 0.0022101494377832001104 -0.18088612188878674947 0.30655624223912131265;0.35244822495990391076 1.2977738236744087441 -0.19192870826580551991 -1.6918177073303191094 -0.62005873082904816584 0.32649141840952611293 0.093016079799564416186 1.1761593423395348967 -0.27102602975996298307 1.4143598013400806579 -1.3692147235592053711 1.1088706233150538161 -0.98748602887844860732 0.024175722571473719225 1.128012937090949519 1.6926812671327098325 1.7457943893190481788 0.5436627244525092939 0.97659166425469878803 -1.0925338012256713416;0.33987294237280774034 0.33047226970258986034 -0.76616862872135627072 -0.046469132711251075196 -0.12346914377145502828 -0.72170696853015958716 -0.74149204183119454825 -0.077793455803771574542 0.17978462571722414465 -0.48608647798115206529 0.073557513220836950874 0.07834683614313368849 -0.28719792139217098503 -0.35238254210956315715 -0.059664439878061151235 0.25816388246395577388 0.12192472468572929101 -0.3405814649127125171 -0.1645379464500365263 0.54857624756732770166;0.22653831815726188648 -0.37246388751385994809 -0.42298052328146501511 0.11148286261136224062 0.32066002872081117436 -0.38396119720000154718 0.40562445108564754159 -0.30122265056090496982 -0.0019595301452835934414 0.12106641691763694868 0.21333363043918185387 -0.00058595556036086966778 0.031019333326685580993 0.09248662296749231293 -0.22200354584817569292 -0.2425474209683261273 -0.19753295746193227922 0.16958254084192006528 0.038178855822318627511 -0.1674297352792190885;0.053238436623896594202 -0.11862026611351386696 0.67212344410579061282 0.20538326935432865694 0.35516722370929204589 0.26937844923866571811 -0.24165606613409110204 -0.36573022205811661633 0.17498150314583155174 -0.39943765553019944248 0.26206701091872997011 0.02903514291063894609 0.19366106713891301228 -0.09876812541365841136 -0.29452224263486159739 0.027821879792535600395 -0.11162474659481258499 -0.39895741253064725962 -0.19141937371049061456 0.35171094560927740336;-0.0019151510597730700934 -0.11940551716521219383 0.0099814788177018075321 -0.13021434729974040723 -0.12655414568389988794 -0.1820313373122383338 0.045706067708955716578 -0.17629108119582848069 -0.11546718331385794065 0.31411366048828931463 -0.030084587744531387371 -0.13026714195800376261 -0.13284568509308780637 -0.11710400473913475106 0.1126809659346855591 0.14468140546879812436 -0.051268626821200861954 -0.16064188814002461347 -0.01894245392869312386 -0.13452654604441144359;0.10045973616477765766 0.028037750842490747477 0.015262559744876348297 0.16814326471099869109 -0.50597076929822437652 -0.15051791547071324828 0.057492752399824806342 0.31764092897232698576 0.13723338587279793055 -0.077230836757194831765 -0.13876738750481870666 0.24151334275186589484 -0.0734162272233136598 -0.0026251288498223340415 -0.16749160821328282212 0.004622879263029342975 0.079206407975896842011 -0.16839172177609204595 0.057300681122215389729 0.04844177392434969448];

% Layer 3
b3 = [-0.46969398725154604124;-0.27475364912044814014];
LW3_2 = [-0.29968963160216000041 0.12213451051315570017 -1.3150486525597273602 -0.88789369119252437024 -1.1538348839183030226 0.83216542800628046095 0.059710404717349993553 -0.077789479648768991571;-0.28023791107027340441 0.055911863468951966682 -1.2657114555419208823 -0.88788246606972909269 -1.256412248979378532 1.0225888094580071197 0.48992101178222924318 0.17044782166067876106];

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