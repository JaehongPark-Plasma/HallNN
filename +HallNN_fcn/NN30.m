function [y1] = NN30(x1)
%NN30 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:27.
% 
% [y1] = NN30(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99978017998781;1.99931525622218;1.99921696369761;1.9994098660668;1.99811214524407;1.99884270226982;2;1.99811404641596;1.9997289478548;1.99839627352357;1.99857252703534;1.99913043677824;1.99861994341];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.32610653160662839145;0.28799049329177511813;0.061442716849255500644;0.20832888040626962312;-0.34427090788508291075;0.031483597698317974944;0.29592307592491085;0.44615311434832133353;-0.87701338082595259316;0.42174226142116177929;-0.2045808628625850889;-0.53286791887000439782;1.5290349638316149683;0.43763858787219406254;0.61154393218819913525;0.08854821964500579623;-0.022546611491282939066;0.42360487503445509727;0.26555139884871364897;0.4830831821470332188];
IW1_1 = [0.048041694519176503242 0.59848303461769281064 -0.39738960937308326349 0.10271956568788627451 -0.20609671978625881961 -0.53845394921549338818 0.90801920765130239577 0.2691209718100869841 0.42967486737817872644 -0.59337869139247123584 -0.069165248039258220292 0.0031467620412956162287 -0.039399019232610535401;-0.10330684670879586307 0.22571073015951920171 0.2409221709575492254 0.30203941977309289868 -0.40116420593313245613 0.024062459363998950357 0.5109097464348096862 -0.25573477357198909354 0.16321079847102051841 0.12494896611773900763 0.236252093407253283 0.024839833437710161917 0.056250855421953120661;-0.22939710743591035236 -0.025422476868789730536 0.45943447747419385863 -0.67587348308723926849 0.033905643915146922718 -0.083705555845120141467 -0.15489458098761346716 -0.095316709441991442908 0.002903141563477692165 -0.0050204963284571782825 0.17046267225693964575 0.0033504124126901109124 -0.011911473393095893189;-0.56049027853772748209 0.43425138696298998253 -0.02147154134607228218 0.011229910738549227917 0.19669896240917925101 -0.34415402537736167199 0.0081385571108663970485 -0.012789527523586370628 -0.20990170088360446399 -0.28635210811776479201 0.041637792910427816329 -0.019225907556028248163 -0.17943852470763821394;-0.54237392415164864534 -0.48514839471938786541 0.2300470236909908095 0.10627793416063094356 0.18815392135211692071 0.45471328639484775991 -0.40206316013895987416 0.0063456532443128300935 -0.15689318797772880409 0.50327546617803631257 -0.063460231875874281715 -0.10130509573399316958 0.074356107113461841762;0.3630558665819651476 1.3781433880042857076 0.19928901676483148919 -0.51421733762755206243 -0.63727257750056387486 -0.46821450681221854495 0.58598362542171433809 -0.097382264091111045423 0.14107384913702691875 -0.44357083114239459487 0.03988365285637736346 0.051260526803836484189 -0.013400855931835608387;0.11721329705658166398 -1.0977275208087291514 -0.13954441144660850727 -0.28315043995314820302 0.73701154849841654659 0.29706420070113415122 -0.072615542272772379273 0.83082499224368100421 -0.026554554264153307913 -0.20574908725555637878 0.414228226925691001 -0.086383161894151030968 0.070564609471077954783;-0.35912023269688642113 -0.51436690297248000014 -0.07718790988495696781 0.0082114263292323956545 0.45776530126282838351 0.37552026046986197727 -0.20093120345048035014 -0.96112775534213035211 0.035322755001192661373 -0.064409767013343013908 -0.025606147620013161581 0.041301657675791705882 0.020010086207151677601;-1.9597996882500126059 -0.73878831038181647095 1.4666106079643008719 -0.4882182220607065859 0.3919480589567896911 0.22062808919931792273 -0.51407700796784538966 0.39575289148911829695 0.064124834264480429225 0.087478207927684875633 -1.5476093756135282931 -0.10983480595268554914 -0.029372295195405712098;-0.80394998738958367568 0.80824790854543837604 -0.467950093047349025 0.98523018482503188498 -1.0844161951783877207 -0.85466290774220043946 1.5501778729370661036 -0.50790118492040958742 -0.18973264102912074103 -0.72602410544807105541 0.19885853397171596169 -0.10438428534889075683 -0.28309918165138775459;-0.31988752955687393964 -0.28190045701101196229 -0.53599117593915035496 1.2250106325772165405 0.13286217764384050666 0.011731157840986928403 -0.40471112218855187503 -0.067721231395974329503 -0.16366475785727627179 -0.13279008814975357122 -0.39540387828018552785 0.044198443199742282195 0.0015117711050422798572;0.067086798408334535893 -0.37752772111024668344 0.044922714274226858977 -0.78485515982467823903 -0.16442410472849286474 0.51686163420904074339 0.018973926358689545413 0.27428191061446760557 0.024089304429842892802 0.51144273344457991115 -0.72679219581712417586 0.50408172951547869367 -0.10158460390310565524;0.92517919636113621351 0.18619661456059471982 -1.0568731091547296508 1.0091150450268120409 0.19960160539429608328 0.050083764749480806644 0.16553118806413769093 -0.56772708366762336674 0.37957504371790573572 -0.31183243822310735549 -0.53397397316345285301 0.33325053146013722261 -0.097633568489782690536;-0.78303660015363130142 -0.098938615288752945753 -0.4006076358497701051 1.3159264159359187474 -0.098578993573204154455 0.057311875500344972023 0.12902971215233638436 0.024245462830124683146 0.10623008658735465237 0.17529549731491519959 -0.24091060191455207495 -0.050952313411372525398 0.074571820473766287085;0.4315563027286786757 -0.62823019058278539717 -0.11241675150249176574 0.27197194257255241645 0.2162332550997831393 -0.091210594235835829036 0.2005021866138272657 -0.22487414132879190998 0.0011219024222032551383 -0.64185743707229792054 -0.27308487800962150116 -0.080889060489687775179 -0.0036457244686781483008;-0.24583924002268875419 0.065391798242093929217 -0.097371164923727840379 0.052986069976774713419 -0.42302337969527786798 0.081337519470783339881 0.56642412157348653867 0.060948367236161281324 0.073009017826199187895 -0.057706573744297273709 0.24259098439080620246 0.085917569044622540964 0.02534419309845949847;1.1339326114750207797 0.31695724593972779415 -1.2296231624953024575 0.32536502905137548725 -0.18924603270675194322 -0.031116692085905228726 0.14613597698365315969 0.18304636860405662513 -0.039171757170963307504 0.33148723222189552251 0.79272349465581326555 -0.031994501973612303469 0.028298075321228594731;1.557196534145676825 -0.18936810307551663057 -0.99466468183994383168 -1.1473075873302625016 -0.12045562815542443236 -0.01450485225549149812 -0.11459659086351925428 -0.42297007878511772017 -0.13806354693524780508 0.20572367597227203539 -0.65978876538130948326 -0.12031585903164469797 0.11288647125858068676;0.26125910846021505041 0.42722266136435171902 0.21773562363980800138 0.043421255441314185464 0.33671224379494030243 0.25131343591132232484 0.1933879185905507947 0.47061838976175546234 0.27564067070253417402 -0.11212246217611124277 -0.24001033386258210922 0.67817423743923110901 -0.21778788949026511856;0.10764708077757363658 -0.50685268414534923043 0.11067901039436846278 -0.26755400320887395837 0.29299905298664574715 0.52753024400631176594 -0.60729676408945709731 -1.8310266342640808546 0.097643431564301025327 0.46893172741634220335 -0.46871951555060403782 -0.054486437661239263408 0.18032492365593785766];

% Layer 2
b2 = [-0.090676163355009858913;-0.28951094196801613334;-0.47180508674827703608;-0.019101755937210109643;0.4457737553972140887;0.5853604563153055329;-0.23239352925393028104;-0.26571624182736158781];
LW2_1 = [0.016554492527312943823 0.016644281258035621002 0.2477636928603505484 0.048741103167925743656 -0.058222938395783449539 -0.20474988691194578871 -0.022693909650443044868 -0.17786344511543270008 0.20560008968151724518 -0.047016990303359151859 -0.058213254904172159754 0.11229562316582936132 -0.15503807258068902364 0.15150991848286224362 0.14241221061605363607 -0.1074168758916995009 -0.081089864113060086681 -0.084949270731092027531 0.022746166302523112418 0.12617845166011265601;-0.35660604551181385169 0.071158631616017098409 -0.4175956953221427459 -0.65479651654709480901 -0.14399140749855060539 0.79633633564712758002 0.67221075794979057161 2.1372869325231542703 -1.2793988133396223539 0.87007746584655354383 1.4334765655382348815 -0.58000362812649275401 1.557009782796338726 -0.48405774141139079614 -0.97528291157912605502 -0.37286310898441188399 1.4300739367544090808 1.2017637042115878465 -0.29659045900155345254 -1.5883475851718871219;0.16978087289128312998 0.027564265114567491899 0.099779748696244244255 -0.61403379389091805596 -0.27182649810388820333 -0.12945500344443053309 -0.24738283085244128223 0.04781353629188550558 -0.041875610669648670459 0.51509712132199980683 0.11418577438448178352 0.0016437170985542049959 0.5467603884255145541 -0.45902566323564147899 -0.090507114048199915901 0.28219259957691850138 -0.079422004127722017897 0.3063611094413240532 -0.31182849276411450656 -0.39228545955909693843;0.26662462009684267938 -0.25013261659399843584 0.036008314961971044088 0.07313636385220302627 -0.1922612851459394312 -0.68447170722124051068 0.079925171425482818832 -0.11721672391765655674 0.057051914154482831487 -0.33785112964252322909 0.39956290346528944291 -0.64243464945047135117 0.23779886492275145726 -0.25579251217387083406 -0.82594567699366394642 -0.44862644689232622497 0.1667217585040977057 0.10804539373777502509 0.26993880856098462218 0.15250469271593425358;-0.20191317487525647922 0.29256262845968600272 -0.68644540637974249542 -0.12569785495052526736 -0.11064840260018385387 0.062336548038833886065 -0.18553865579705072775 0.012028492600363155027 0.73653357076087944133 0.20581116988921266397 -0.19301392163276401281 0.21655763594538413042 -0.076032464553639836358 -0.13121984487752189064 0.35418128651913416416 -0.042726731649805317625 0.19339280762864641261 0.063572092284210096613 -0.063574163014506829272 -0.034674257342333238141;-0.3147802192862369508 0.83332075621994872172 -0.4812425546802752141 0.33354947356238134892 -0.49903796001104505642 0.17201979682837045815 -0.059267675883561604655 -0.0094981955680512788337 -0.1474765941268887659 -0.17004746051643174209 0.27731699206701609928 0.12299763131669982918 -0.16121443094299162113 -0.34654406538587279041 0.47703526463555623316 -0.10019257798395470038 0.6880867941721278136 0.010331186018662313897 -0.00011524549260893515485 -0.021698481771925716421;0.35041276565601103821 -0.080196477718642342203 -0.21712426348589114466 -0.29629051465779426833 -0.16808549929244670507 0.29937897017791953447 -0.18975750040017264064 -0.25063248305673752059 0.067441577092609797384 0.13191055558241426837 -0.088556574795230547092 0.10457121813948311306 0.17572097807251996215 -0.25760107237801027624 0.26552312394053040334 0.079713890371745019237 -0.5523494742953233283 0.026641988671916975312 -0.12197816855445485495 -0.19868525769014727667;-0.096108724616435589616 -0.45001989905447664642 0.16762120448220510505 0.24776107144607392718 0.20063895438782775038 0.096587276061208607314 -0.27594647629591390059 0.3382321697832165186 -1.6408493391357448221 0.12607105733200901776 -0.10354567534695259534 -0.16474451627860159375 0.52797190122558879111 0.44750227757375432569 0.22724113746838378591 0.29132267355628255201 -0.14960533044485324616 0.42948978005373239508 0.091011148827987525811 -0.29262896884470729342];

% Layer 3
b3 = [-0.13632873304717640917;0.041005484335301677079];
LW3_2 = [-0.13393755579959612523 1.169559325293283214 -0.17869211646453414533 0.46699813861203826049 1.0237613180805607005 0.7793820520133200036 0.3881272964885322807 -1.1535604999557473871;0.33866963453594145506 1.1446410415480439138 -0.023302817711234876524 0.41777263563762229159 0.90818280163781717196 0.78659435102783092919 0.31033677045462915167 -1.0554307678281533711];

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
