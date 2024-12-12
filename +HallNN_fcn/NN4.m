function [y1] = NN4(x1)
%NN4 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:26.
% 
% [y1] = NN4(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99987160518484;1.99915641082501;1.99854057068497;1.99933076260362;1.99810911901155;1.99954405278841;1.99970834236048;1.99836319699104;1.99976641369419;1.9988347689939;1.99803687814049;1.99881563055689;1.99984616564086];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.042545305223099315339;0.29381050712880985021;0.12183446265161529398;0.35476772511283583;-0.099113053363040756749;0.23358694417271436139;-0.85579796634659732746;0.19177809464098505288;-0.30423792155337081589;1.0558890388294202012;-0.0014408391075829635533;-0.23736646224411495076;0.19435493427916095133;-0.1209092594343209176;0.58138765384606538866;-0.63461963336568838479;-0.20861336133855828967;0.26364797876565254642;0.033579101689497972694;1.0047967986181292055];
IW1_1 = [0.28627449612402672274 0.6432950961064741735 0.027385676626094757058 -0.07759522196651395809 -0.43307330770548391197 -0.43672837710589307303 0.5533532710685872269 0.18946873449245488219 0.013992652723525166383 -0.43108666027376890684 -0.052487248463688331024 -0.034501810305217449348 -0.064848779262701952697;0.24512134417041825385 -0.043279974256678101219 0.068693354583782492706 0.34246607568644676922 0.28099293190974894419 -0.1246364419560682113 -0.10091398664835450616 -0.019011608977822277294 0.38584849510412028151 0.34462613456521556943 0.16835066014103211574 0.055573652520366070839 0.073464833490823056028;0.092172722870325060618 -0.30823250109636046723 0.04556008297007795721 0.32903532868383172927 -0.28469372383102775625 0.21512798694387522125 0.069791353352376711561 -0.2169989364084760064 0.1632115625590925545 0.050285081352932865795 0.026398928750160251949 0.059581835357807165499 0.0067615562615184789108;-0.71038059948021781143 -0.41399193092662583959 1.0332439169077429764 -0.30070241835719896395 0.30642030407361575683 0.28480330397080638738 -0.41190707866659070913 -1.6223078949332607746 0.10121425533731578938 0.073787253369406335946 -0.32703818395344297354 -0.0044980593005084137243 0.069670192714931675204;0.22525504889846428846 0.096103561861048319992 0.0057851601593572150525 0.028275254549050901631 -0.10672223841078921913 0.025409912811765363039 0.20422626744957156353 -0.11558311058101422564 -0.00064125541238928469336 0.003851010008842489004 0.18204265591372528754 -0.013627435923017796118 0.011629328751087936178;-1.3709397656332491522 -0.30475225494651231317 0.22332003447167816934 1.0080749726057687266 -0.41105199428844563236 0.10488708689705078525 0.43301991405070716645 0.089968280203678141493 -0.028064928732064280126 0.15619964054931892439 -0.045511324819544361764 -0.052560864299577521319 0.051298345749468426957;-0.29824691630671518183 -0.85794536731169401822 0.24418562530062398741 -0.094244619275872243414 0.9484461458949474677 0.48623635281947075182 -1.4439045790623747934 0.64917432128593188789 -0.016343126747864897969 0.75151396367684619726 -0.13212741880207870548 0.13123186677674444911 0.20638821358335926459;0.13213055354694358545 -0.2018322114869816053 -0.27266083220140524546 0.64814105082983697326 0.22744400156004068725 0.012190671428720907879 -0.26028110825277639417 0.15272166357557698246 -0.014051891156914724651 -0.10706870980446563102 -0.41733114549209787691 -0.007591798634109032072 -0.024021879331646938532;0.43994265962949419269 0.054935632042894859406 -0.038186670687348683861 -0.20315527911859537769 -0.60604783892171298909 -0.19437179250218364346 0.41205796624743368239 -0.56474125510551276275 -0.086307648425225161337 -0.20018319603783429894 0.11839043280130968905 -0.067837106744564543148 0.051825533170594476062;0.29415380609929381306 0.59095011491170923446 -0.72116218209975413433 1.0510381157170460842 -0.032905940756517336465 -0.56073207320547224786 0.61060389107825474575 -0.47857287366835277176 0.32112690141826188972 -0.41847469282093202603 0.13032779643589303964 0.24422241965818283771 0.16902478161450662508;0.51073911029385321569 -0.10530409825893276288 -0.1328693305570414207 -0.34547972149504535544 0.8048634621161430136 0.85821432399564467897 0.27634418830606943818 0.11144285815849493126 0.52402008220433682339 0.5070507952222670367 0.30525739001567564479 -0.22040097664011648604 0.32447890427017200832;-0.71486049440581811432 0.31993511400256879895 0.4876050131718497882 -0.67951762807086313334 -0.52419900608185654445 -0.86677524250211479551 -0.37323137132596506671 0.11444430409383843372 -0.18483017598417433858 0.0043786854547127156134 0.13887937707627048822 -0.10932743581587606174 -0.083752195367773465962;0.35293369660235263119 0.80942984210183943627 -0.35138507359529874696 0.15388869477537206465 -0.12186577588273209138 -0.15623924197786467905 -0.69313340099047060505 -0.0088833424849654816363 -0.14394045086043294468 0.72844126629875083356 0.31574731380934539793 0.12482075723100491405 -0.037371078660202866617;-0.033060397232435784209 -0.047699099245545106829 0.074293593754033424026 -0.048382834290419368672 0.079431851785647614173 -0.096426687797468532692 0.19873027330344042451 -0.45142727690831940812 0.065854612889226046835 -0.59359018126252838243 -0.014077030433278424348 -0.065909373488518566142 -0.010831044569041766573;-0.050497965438991661291 -0.31305316930124499342 -0.93299748354316192067 0.53479502250624677551 0.19382367054258553551 0.61796458247189489654 -0.53643598548003157411 -1.0551220167031696295 -0.037861379598323981199 0.36253137022825965285 0.44895178442426447729 0.031652168688629411797 0.059785487200188967838;-1.8094167481336904668 -0.33916109526889454129 1.4229231469655150288 0.91260094894959575829 0.29242090467739695336 0.23715263771003053384 -0.36013158667038525085 0.30790399330448159354 0.10628447498786894321 0.062855304736068443039 0.45353220555608719788 0.0037886779820048475075 -0.1051769657817067799;0.025892928633159586344 -1.2024112868376468644 -0.11661628452005563805 -0.081619335829461014953 0.7982172849137038062 0.60735190973267849657 -0.86828976869215634959 -0.17686548634338838948 -0.086674470040482093358 0.74567289414216020305 0.10020163871566722946 0.060194766831389724671 0.056849713796150627698;0.24596083172138200057 -0.25231151374485749628 -0.3806122606354944149 0.85128473571323259872 0.0082637885397042684654 -0.19201540252196097613 0.49484520040117480066 -0.61120914782864321779 -0.11094948518886976752 0.04319455571360726609 0.38663094493897559589 -0.10327467706142445802 -0.063753435169606520461;0.59140176459347593152 0.11070677411616493269 0.013302804425315674711 -0.93621387710342884869 0.2458078177519692209 -0.11603297715223759301 0.5172969322595952768 0.13638797455909870693 0.088581903625905428212 0.58984247518004861988 0.17202424672349389545 0.27575010313341546953 -0.14878644932118301192;0.40780973510939888538 1.152568188632921764 0.54042496401261874706 0.23703213969127800986 -0.42008022454751436925 -0.83848483227120318428 0.46531977840665628143 0.39186996777915633627 0.013879050863675794492 -0.22954601083491846247 0.4024769866733125534 0.16523507682346416647 0.12024101403116727038];

% Layer 2
b2 = [-0.790137297450062559;0.18875903467944282754;0.13088217080857939001;0.77818027658408917002;0.099867692540663452472;-0.11297838441450490377;0.32564487629775074051;-0.17676112779433042488];
LW2_1 = [-0.017206859349791971125 -0.58778067441182979547 -0.22638860316086981128 1.7339757110035025622 0.59250114970175116191 1.2792647754484887113 1.5991693316680739567 -1.0789010404196364146 0.46830259558165859834 -1.4662300307042051006 0.3999837652188226178 0.68552476457119526643 -1.2103964389110184463 0.69029255876032535877 -1.6075069444937430951 1.4888510399357905989 -0.31728269079350046589 -1.0392050329713866663 0.99553087387898919758 1.1205863439697831829;-0.24149167520324468783 -0.14452478412078031966 -0.012438714545771594605 -0.033909974532103509537 -0.17182050904336693131 0.19103792764006657934 0.1236162127827938606 -0.31618307342070040011 -0.036273069621012510311 -0.17529264651178314005 -0.047524733810686788094 -0.22217708769397456758 -0.03412202113513500279 0.25251228127253372291 -0.032077519723613417968 0.01788883342262911208 -0.210125674670921192 -0.10082150258277211563 0.010332355037352514843 0.4384941697033038488;0.3937900926436587179 0.077761560260634299269 -0.015944605767056790196 -0.30275342308712555051 0.28854170537548767816 0.14248716373104480803 0.22804637387079695898 0.10475030106568909705 0.46382794366705670797 -0.18282422031691494824 0.16568814926696423528 0.31341505865259872143 -0.59499673491668392256 -0.15656486698566429716 -0.36304477859952055674 0.18926953356822157071 -0.32872933383846175692 -0.17399756846810643385 0.30526117709251121557 -0.12549432615616115672;-0.063417285377182178174 0.12977265429968234955 -0.6978311483397116044 0.72298442456907263676 -0.36651699133065013259 -0.23136932080179953286 0.56029538584375537802 -0.16619923632056138318 0.84405674488784432352 -0.16926547191528751313 -0.33331919690872252593 -0.50495656130929067462 -0.1881800765425970523 -0.34829367280464623713 -0.47487427099128559638 0.34600701078666001864 0.17294123976200925785 -0.58054367850909216475 0.12677182542128484499 0.38103713451100518217;-0.081329415832218629467 -0.018425162243100895454 -0.25252177925537194758 -0.14260813033937178029 -0.071197067641120118098 0.14226990653181739721 0.020530642940295145554 0.30337676160885063137 -0.13868987615487224363 0.10808882665499518205 0.068715107441446890424 0.39167590610140240459 0.35237985097139357116 0.21341100280720637361 0.36871190493379724096 0.50205543882783154608 -0.062966649023951853925 0.012186838031476115302 0.11452524536745367634 -0.32441073897514549618;-0.29184467989069673433 0.13873998888051780609 -0.1597910556382263203 0.1278357616258315177 -0.10000131619759079771 0.031859115147921272804 -0.20340509250149674259 -0.027780554961978495832 -0.28615750409460766202 0.26010334312948896196 0.022502412742169335819 0.20569823349960184022 0.0053542659344838205993 -0.18594595135474342729 -0.19924006098371802032 -0.078390671063616468306 -0.17743364548347523302 -0.11090809487344011064 0.017688228164748263926 -0.53663594143050397101;0.32403023052586682473 0.095320089975469160515 -0.40425860296971533048 0.36405164699216630009 0.6194223526311135597 -0.26923630878680943068 0.10617510711894162512 0.23079317238900448617 0.046110164578369292054 0.085858660233106814941 -0.092742525692204161314 -0.1082252336408083826 0.15476050495637280746 -0.23099149800652790665 0.14615475175655845419 0.090261198844071216518 0.32929204861589927988 -0.20319979658952391777 -0.064227726796967560263 -0.1604273085493347073;0.042492087949165549543 0.037592620307048012218 0.098960581932209190636 0.20086865476802159836 0.47891778022063985487 0.1419773725449670343 0.25937925664545313786 0.37485011261968215202 -0.13357307578060489961 0.0192831731832280584 -0.14426136040644368941 -0.037280870472562219198 0.0094053939896559732303 -0.32804197772139825906 -0.27246036961953756794 0.049813198705609913752 -0.21139414125077826689 -0.1799385630290305238 0.13369263779986353957 0.1442290158749243989];

% Layer 3
b3 = [-0.62151665455962767926;-0.4203195214574402816];
LW3_2 = [-0.90057073370420082092 -0.32708306227026945345 0.40571305673083662979 -0.46378668781323761916 -0.29152610494687136722 -0.72287662984426548363 1.0056275064743180891 0.51139439887134163776;-0.89387039159140524447 -0.65736253958964419031 0.31824090895406614843 -0.24318460765827742964 -0.23166527247778478937 -0.75830753100938919076 0.53589215727675809653 0.38855868591351205854];

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
