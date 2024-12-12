function [y1] = NN48(x1)
%NN48 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:28.
% 
% [y1] = NN48(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.9996955012233;1.99899072162953;1.99996145554687;2;1.99816237873712;1.99901510046506;1.99892320071752;1.9980813971049;1.99907681098659;1.99804675500198;1.99803774563497;1.9984907056677;1.99949519774787];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.096483099844948405632;0.25595790444064853153;-0.12856952977433425134;0.19205220762283908842;0.12356498001611282422;-0.01936340542552821356;0.27479148949640996857;0.44191622678578879047;0.24731229859068937227;-0.11764345900832337555;0.48169018754891312906;0.58028184957433659452;-0.2925826519162261774;1.3539379563429376763;-0.20424697615410894458;0.26721908064742461164;-0.051977408841833667996;-0.16910229609188578981;-0.68472586376106114869;0.057733521562689148199];
IW1_1 = [-0.3224659450213747669 -0.54167313447675802252 0.71826240007643316066 -0.34448635795412807514 0.46601540015559156371 -0.092154517434886853988 0.16837927855451478854 -0.3561591092738037867 0.027920384871084759359 -0.70209334967679026818 -0.25859820235323421889 -0.041944058375790585491 0.040366553597321788716;-0.39965911673163456719 0.47801782865362152375 0.029091016605682772089 0.34589400589831753585 0.1877187952990627684 0.16493528617673125503 -0.63363387866004750038 0.35427880132233557919 -0.070365164614453870606 0.33775272776505060879 0.078724109838865477529 0.16898022446169908006 0.037198663979239744715;-0.1726354601449789361 -0.11952766505535472596 -0.32337936962090502169 0.83978083527280300569 -0.10473932999677654232 0.065569053847285271863 0.22222575950366255104 -0.15712372248353673188 -0.038047761581273022158 -0.0035908941468890303098 -0.13360424600946291251 -0.028144664333736302275 -0.0031905298337398940722;-0.15531005086886731648 -0.065968577980488246615 0.41852719510677477421 0.34486227628678595147 0.35217629900527208653 0.076669271769583741127 -0.18484788775973098884 0.33364587654388944138 -0.24536819687288544123 -0.45609820454489646835 0.32396822513318379588 0.15064682420421093845 -0.048106222504577078602;-0.43836493017958572116 -0.072855739447563280797 -0.04041544828380640858 0.23801764466589608871 -0.38623221019588216407 0.036734612887291739902 0.39169364875215462174 0.032624326936780058406 0.068990392332679423282 -0.004464165670529269235 0.037722770095945851176 0.015430885361908864076 0.0309658286682825655;-0.22365531402537561223 -0.37883586367672689166 0.15842205247351870412 -0.0048710686892251909078 0.53340607954265650825 0.11577935911546809178 -0.63006259064977765405 0.13119094163840125389 -0.068370106622109114158 -0.0013885061252801016857 -0.014484927840219560466 -0.010001538640831164964 -0.04008172831304405298;-0.36959257504485792944 -0.29242015416687805462 0.73561406890155311977 -0.34246265463243247273 0.073756714924326033378 0.38137374508750188085 -0.30534900232548090049 -1.7058958245240445351 0.11012583472719951472 0.2869766331264290149 -0.15857716315548836539 0.034719446635994931361 0.054272158596962313815;-0.19739654050441030053 -0.25469864306170936219 -0.34018092453617443383 0.64584358618941617447 0.073528687539865772438 -0.74084302465671403315 -0.15011889214492329936 0.083282315212871563248 -0.11160667013432566674 -0.094180007085280625923 -0.26732036538746195564 -0.27080774346467789693 -0.16662918648785801645;-0.020132224172142592494 -0.45966610280952757694 -0.13694542899492476162 0.74543228074924439142 -0.019774779116652378003 -0.24711805809132797451 -0.23784078928681864906 -0.39863146554308404035 -0.21184335097458706354 -0.15320845934662724597 -0.072026514546261632965 -0.0032310544658845459162 0.02046591271815358104;0.72698600397223678282 -0.29743384990069821061 -0.46654569138978424503 0.25506733512110690842 -0.067553180502974360944 0.32222380069660844226 -0.18230279702904814143 0.23420254206073864989 0.210800187300574704 0.10092965233746860332 -0.087010159341675857858 0.015376145152782346587 0.062496263386260200723;-0.20238472349380456916 0.61610845693420168701 0.62256105037020492698 -0.44681179634640549869 -0.56509544672600109383 -0.42846616916513546869 0.048896507805570019189 -0.025436725370547103725 0.05070070145938548184 -0.019865858531767179029 0.10521348467928193227 -0.047267726000901573447 -0.04700158401690973653;-0.40671354811869819823 -0.60762051872947897557 -0.39801305954083798788 0.12846783303041175994 0.35608282495906784515 0.14127439878574096777 -0.084298714203021457569 -1.0716022613663731544 -0.09228066183317758242 -0.45161602001925260641 0.28942390033527437554 -0.065788832594639815032 0.076392307226480257731;-0.62244789138235856463 -0.43118064250668014514 0.22553610201695939441 -0.77047237348463004647 -0.25133419343287599634 -0.18956414814584826578 0.04550399409787167504 0.2599539718416483125 -0.55480824881187285058 0.22317758584542099487 -0.30213615658130366093 -0.3134953536294467491 0.11271299092841674527;0.36564769776191829997 1.0353354371748679785 -0.49747426096121494243 0.18591485209927346989 -0.75772155639714844355 -0.66700060107179237612 1.3063169383880472996 -0.97956749257086195293 -0.046852766545195718983 -0.85723547898395502376 0.026698139302581119087 -0.0074696055655886009483 -0.11544038585086856263;-2.1126547157152741896 -0.038274612743215107036 1.1677560776397026832 1.2786010227299915432 0.070898537563639688064 0.046558186774962935506 -0.14397816087327058865 0.19224501221937040363 -0.032399951322322380343 -0.006327632132975831436 0.27386730540017761859 0.069287626629901538644 -0.092988882920580662184;0.49156917659536514265 0.48794398112741110829 0.34004092708087224617 0.32226106694101391836 -0.056950569609146156624 -0.15369208892865157678 0.080294306610742716135 0.33768241748727867435 0.14578457759143895189 0.13555974030921064122 0.30663337883132757433 0.064068837248137194829 -0.051566821594879436175;-0.41068558507311314631 -1.323646046175592339 0.3231222363860201674 -0.19695942202168092994 1.0096535666650670038 0.79015811732998375572 -1.3899924764329736337 -0.51658432588884595216 -0.082551811408942521453 1.1478471790704256605 0.017549155244399636255 0.058428871037293342305 0.062505712425568821233;-0.27142735538584916721 0.47036661510339711212 0.36192078701900848303 -0.40474810037479397318 0.023600616106035459535 -0.51603610619239337343 -0.12955149866432227879 0.37232621253239939207 -0.18357144321301341328 -0.25682705614815343198 0.3485770420928929747 -0.089391315318602823647 -0.10779594201911628371;0.061064983787014352756 -1.1550928599405079211 0.17259403737101872145 -0.40097733464430473926 0.49234173329407288078 -0.015099184244287831153 0.03107203167065356042 0.86297404374708353014 -0.14073941442959331249 -0.56652886947640301862 -0.0049413970363541377326 -0.04843705427281203707 -0.0170346820726948317;0.026407928700510386461 -0.071332255108271971911 0.48725002515339865816 -0.36473843456994037293 -0.23077311508764714043 -0.19554615488038126769 0.018225812037306388103 0.23920958591946672733 0.12447627283803804066 -0.14660163155061634033 0.0051405638049893754085 0.04981661660488331117 -0.050535106346645423947];

% Layer 2
b2 = [0.34583143900192581111;-0.082444504582071961862;-0.94523108694789093054;0.23153879572898541928;-0.056928927776702442642;0.03400857971614761055;-0.037258695355764287482;0.076157967301451551245];
LW2_1 = [-0.094677882586468073423 0.1745307113666709764 0.19036172854527683151 0.029985445403312219886 -0.40155696943492030782 -0.10415376941670022315 0.097926271696878508521 0.24573014769457873774 0.054310163327146873669 0.19526926513404124774 0.029783925969604697903 0.22925678926972503646 0.11017122913203994394 0.2028491156665933326 -0.1950396809637836848 0.1889962308586368156 -0.33153426414611203432 -0.26312983247139509757 0.17114163208223678869 -0.25547830548095395997;0.021238741220415875005 -0.056478378011995560015 0.36212754826323395596 -0.0092653097163878928022 -0.36483442819056582929 -0.11252487218626038012 0.0046641669197685312748 -0.022564506953309165782 -0.19887027164914330979 0.064994919792729519137 0.018550800278660496689 -0.18429744269705064519 -0.008072299719077091068 -0.30162156424588126846 0.24769951827831482305 -0.045061988505880203049 -0.15071968056803475289 0.17032952659130493056 0.011606327991223571588 0.27059160350474081858;1.2451970091096009607 -0.67588662893965090195 -0.70514962969316929975 -0.348135931583872682 0.21181352206164119623 -0.13970920475807269145 2.2696493345083355031 -0.95409702004833496503 -1.0350696654952313835 -1.0019181705099311497 0.98270828064457760309 -1.5598606327862671872 0.97180792354787692844 -2.0771852776426942278 1.7067756899440851281 0.18399821008389147248 -0.56656283671207174901 0.78548397290985216834 1.5242230658203939075 0.39626210200833233976;-0.080390183124516220725 0.18811121837065403195 -0.28062032132474262758 -0.18589658218738788764 0.30793280689618457435 0.44752605709530068889 0.044911610127302076245 0.096849032710983926076 0.023987366819702898857 -0.19708045184044856168 -0.18503550997512754894 -0.0043189816411835337331 0.15079977226939578205 -0.18052037130909351359 0.18867972603682148236 -0.040257515263995166666 -0.40710487728951022524 -0.13464397198258418187 0.1662763597424416484 -0.024321407654413072091;-0.35891319580318292726 -0.62766386041991273359 0.42516209167214147335 -0.060492292721806457945 0.21875188197743239416 0.21064676030568965648 0.066573453121828940482 -0.17606993803658282927 0.067792439595529094265 -0.012152701356751371628 0.4332797855087806016 0.19586333980714396019 -0.064434870690493459278 0.4442769153000886706 0.082591591586905724331 -0.12411309858422821251 -0.65486721349570986561 -0.42285983600559612361 0.151569011818149052 -0.051972975224509304626;0.20498756603181553237 0.16575736660760251318 -0.2046816852602165171 0.16514739632409183856 0.1649365268521214567 0.13916258259579181167 -0.12902242985377607165 0.20617425506650174927 -0.28931186759265747899 -0.020019709920706216244 0.23181137963664519286 0.028256912323295114986 0.092808341631826052343 -0.16725763624970696464 -0.047270734459858639032 0.073283804766020033128 0.62929034337766387441 0.0024006037876028842384 -0.29446938368558983612 0.069223890352255484926;0.021675547875814854537 -0.447690281468931639 -0.21384341290501540644 -0.23994934752543228829 0.086148528098332288749 -0.025650454668635486871 -0.083031639264650805576 0.078576484540735014317 0.3757001617684934125 -0.071059107272163482549 -0.082526761209725504886 0.034989944111764162415 -0.36770549482144654352 0.28793809590078833782 -0.22422283504874757076 0.082545771413029309094 0.077156895164221195316 -0.049778609745889879079 0.033073578036248982293 0.22122122195457963323;-0.23516332759225097582 -0.087542790363461617242 0.21436034526373898279 0.14094439848864021014 -0.38748083451255305176 -0.29257735119498406995 -0.0079082914935543983243 -0.074832952553512424587 0.057720016083906422466 -0.004539221013388658639 -0.1069475426599977852 0.16242687209067985243 0.31330518837985715797 -0.055478752862298481852 0.00469928731740682816 0.15272395943129352114 0.16802396218620477275 -0.10474394525967695158 0.0029328239983265076002 0.31782541083677079419];

% Layer 3
b3 = [-0.6125029373187155235;-0.5127379821291139228];
LW3_2 = [0.48395418253439498324 0.81091522377828073598 -1.2219133347580040105 -0.75722570343557260575 -0.37439251714255794568 -0.60487912576301183787 0.054646514135986125338 0.48816024553374082195;0.36097575669054848824 0.78592101757791854588 -1.1948758389681715109 -0.5366857513666085655 -0.30910283878448147066 -0.51915846620566563629 0.21722566680370930525 0.62578864944009537918];

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
