function [y1] = NN2(x1)
%NN2 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:26.
% 
% [y1] = NN2(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99887176025088;1.99911388175249;1.99851224171557;1.99916966278258;1.99823865035276;1.99956614086874;2;1.99839266720095;1.99991748582529;1.99835561071971;1.99822257796184;1.99881481164286;1.9982689754742];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.058805214705198345648;-0.079701570188269829642;0.12630393275803936115;-0.036142312019772027121;-0.68202424597435629128;0.58744068310717012515;0.37570517810063697173;0.77167713218528133368;0.084595392527687274242;-0.62733495668583938443;-0.033961619408276681742;-0.10344244030806930279;-0.58624653252282776794;-0.84227415462312293926;-1.4795623553602985645;0.49054229342092470212;-0.96581776128224827715;0.42240866870792481702;0.59071855812880880165;-0.4349141908663338163];
IW1_1 = [-0.14134051451075890782 -0.1088238616797491537 0.30591390503231619125 -0.51010482692208292121 -0.46876932765523193769 0.20350711392205664274 0.48392591860169764661 -0.26197283832899825873 0.067235937248827148949 0.31800448491085003822 -0.29774928728950400547 -0.050672796835911547431 -0.0046470813085239016937;-0.18738279280097974722 0.75634351051338388761 0.77612190649530632403 0.86109635337442658187 0.1685412392128547987 0.2930285820852472356 0.56424404974134312329 -0.22430605515261359262 -0.30175071105764300627 -0.030660188729951125264 0.26413915696845180481 -0.66985116882492823365 -0.6761007292710049521;-0.497486047591858771 -0.13697848216121358278 0.079137272181751497357 0.15842921836299447835 0.33258817734305251568 0.07812415930357216598 -0.43740260154518451063 -0.040677827397053369085 -0.019049403154916343117 0.10079749470063661954 0.00067928499577905442509 0.0028246090990733463758 0.011529993737164938999;0.10002863923870110308 -0.0005109400446696036116 0.053060427384348511637 0.059879365045601970485 0.57156762744386158115 0.073138291754202999773 -0.79832459096321661818 0.034229801171863520615 -0.024269085944210025407 0.18021539622798532498 -0.073374022144244155363 -0.020166117123095593511 1.3188580009906554838e-05;0.69833066902341223159 -0.56973806271357618769 -0.40582513719681984066 0.57744859922766367255 0.42537919989264394438 0.99790078140045856969 -0.095529476320826575697 0.39160389726850508563 0.62744797455378698015 0.50741289710156034509 -0.44439080298830452298 0.13483285483363754986 0.33650150246263033083;0.50826856087836769227 0.29958915565943705639 -0.097134120018157410481 -1.486064612455028966 0.12654268053059025689 0.023419457002538522639 -0.096596367132452412574 -0.28264185315778517227 0.031402196012084909738 0.10680390303941324215 0.29758734748141502102 -0.051212781605056434397 0.039293838227327307355;0.29862844996543463916 0.0073101487691153866494 0.17493171013736522057 -0.43104509183746203549 0.29018359804424664894 -0.084622985642777814874 -0.44851238408437194005 0.4053228318063508695 -0.034366362384410376052 -0.039831156683048464517 -0.18938522427301171525 -0.00098348535320662896358 -0.036286643831142259553;0.19284733641439830154 0.73122175978496140036 0.11823368979380607935 -0.48625985274563821026 -0.028477014660345804226 0.10762311957871216472 0.26737796811518377282 0.14107314186788755261 0.083155260378498963014 -0.27840617371848164208 -0.5887868110517171516 0.4211373606120246138 0.06774092862733289977;0.60325528206938172637 -0.17474979610958193965 0.012536984186660431184 -0.37078084969632313372 0.22810836185853328173 0.45305245562064716536 0.68441163107982383451 0.60586753597644826819 0.37823007685106119213 -0.40392619314528638297 0.47738732831063901418 0.067579011919772349892 0.036895507313177450248;0.25698519623092663799 -1.0135885548564811387 -0.10688408423647587087 -0.24722014021553442853 1.0652353039735902307 0.48809122163613066769 -1.2248224056837206941 0.51847800968841450953 -0.072174341839391573106 0.63134059729496661006 0.0542211049876513973 0.0070511649578819494966 0.1198680725933806146;-0.49383296002519960766 -0.28513680533988577936 0.45927702013866300579 0.040437600845103713332 0.38816940680225869853 0.31643647323321599796 -0.31892965518056576313 -1.7454863554051447405 0.15420887778733480267 -0.25550742959467664939 0.34077585197923082605 0.026121467450140954669 0.13978229438289771491;-0.72371634571042087192 0.081978868674641755532 0.24622645317453006886 -0.25602787880011323995 -0.069543548152131054452 -0.13519630839426866054 0.15379663891507105999 0.4120027493967800547 -0.021347121070210944849 0.34392741766901779865 -0.54357815268310982759 -0.73481292224957206471 0.11241994793565458777;-0.40319032687504907653 -0.032417014989675814463 0.056932356411884960101 -0.017907005059648470335 -0.2258037255969257695 -0.5285339994200207725 -0.2069364540755047055 0.19556869958187014791 0.29892733254723113934 0.13173895822341605277 -0.62375944635080438072 -0.41495333205654133524 -0.029690786522555006471;-0.25276791767023071289 0.2625606431591696599 -0.61137085399824653287 -0.12966780983979203645 -0.086536752619993176361 0.046825028915841886146 -0.073144288337327501015 0.47869467747568256177 -0.07863152513382078157 0.039680978703290924192 -0.30383610579141545616 0.053295476798325844281 0.021526710021184086719;-2.1245085141279602681 -0.89549433107871578841 1.9456177074971621188 -0.66128704036549279532 0.014182579424477169405 0.5191921975002056211 -0.6300439501940702236 1.1426429479900468422 -0.054899187870077040263 0.85070469205115517575 -0.86060912747224338482 -0.16381624129567681769 -0.067792023024507347406;-0.43179642534117873565 -0.54286268218878674929 -0.28654226236653623783 0.11100844303877650809 0.45587756960962771968 0.60286270072999093017 -0.9270446688382675271 -1.1985300969900776202 0.092127198458544376503 0.80530114348314796313 0.24310382460520851833 0.19505893361974305233 -0.023681681278585123285;-0.76622996868294979134 0.22345406709199408524 1.8186310850430273955 -0.5748662968746011126 -0.099736756232514531262 -0.072901159249967026343 -0.18712652983136057738 0.44843814664109937773 -0.087426849415384658148 0.35396552124501756698 0.39683352274225613954 0.041512675775049613647 -0.013417049404240707217;-0.66541282543388835524 0.04707698619752461594 0.7627423139828485299 1.3510276207529405834 0.65846413309598705865 0.20074038734750299096 -0.29980126127813389036 -0.19655614396130166588 0.23154825687756361297 -0.05445845771674980712 0.66146083261793287011 0.13629305461612603589 0.082351574104345473537;0.13628724409376308246 0.038187896552954014862 -0.44666756707418808769 0.54891607654617158296 -0.55876279952199936485 -0.67329016581769618455 0.76453220156818346709 -0.42410207472048466748 -0.20154972833416798239 -0.55157576685233944414 0.13417639390695879076 -0.52474489611778440956 -0.021321935591511277652;-0.24397529249927366046 0.64273666627215253477 -0.028441262722771028931 -0.041670915129482649641 -0.22067278499107123579 0.33639845185599542843 -0.49487102213379108306 -0.027557596347244594365 -0.092375351938997846135 0.96800617030209468172 0.20380542321456182608 0.1138070928221835143 0.036959640775990211525];

% Layer 2
b2 = [-0.29603061003523245542;0.087375601248687165268;-0.10038270668876413116;-0.16922201339452214164;-0.59590723942223655651;0.77459791245809717086;0.11146745079973716108;-0.30247011059949396872];
LW2_1 = [-0.13242750405831163474 -0.14111539064772180008 -0.32103094810089793931 0.10257833195509610813 -0.15709288576361304934 -0.48227349094843890986 -0.51570149967222878029 0.53646071046773446955 0.52245706199678554338 -0.21782353499244724859 -0.58744452656330037321 0.14132142465946068155 0.32399962105987539784 -0.77172855660805528277 -0.55003769673118263484 0.39255397205825370843 -0.66062591291049799214 -0.2783542433672919203 0.48712897231183172053 0.51434796104497060298;0.25910135275753465312 -0.14754931300983309406 -0.50036518932021001316 0.68372059406496965561 -0.2760907160994451881 -0.56057598161899446776 0.06235115320872768041 0.73595375790652606263 -0.20958236264395652126 -0.44465787231130698975 -0.066249152839114908153 0.41614589564070464389 0.15820211182399690486 -0.51803541838195377167 0.47912885166450791541 -0.63545357613436159738 -0.77833003299351199988 -0.41358890714923213228 -0.58517664592872020712 -0.18016115249105157936;-0.59029077085787717838 0.049239812073322938857 0.2327237871740516495 0.51305813054890647695 0.19775257393595471744 0.39733676286057201255 0.26889931922484677784 0.068388366588248386546 0.41049343789332332078 -0.31049739141595311676 0.22334552125093762309 -0.069901073899642854337 -0.20759188101659134529 -0.18047138300438286818 0.8206358583859046929 0.53173671845096082134 0.022135547092706549244 -0.38594726969937348526 0.25847089271650702802 -0.37142857987366195704;-0.28304035327713633619 -0.024201864136426783081 0.091081772097483096329 0.42548209257530944516 0.13529567291376565219 0.25922409257621925605 0.0675745027011966648 -0.13306733261266506796 -0.14700459840598778527 -0.12825162093568584498 0.49437822085629545832 0.342627364808407886 -0.63904467168348166872 0.39396704155517453527 -0.064480093740670971214 -0.18027612351733740548 -0.60817272958429924312 0.17478375203663257254 -0.29528584923752654312 0.41013139426244343833;0.32805373762905115642 -0.18614672403522877775 0.61387226831980135078 0.0008562732585519657047 -0.26093052609222466831 0.37733852510824855697 0.39516026370491369457 0.071213153076227900251 -0.20893390787741836223 -0.18356411723935006641 0.63938916233306586356 0.66619717535242839457 0.056671508600271418288 0.56423325332526252929 0.095138450393671428196 0.14058434435990474487 -0.013057191131181396584 0.10216420311753564298 -0.36326709758592995447 -0.31497882122878595457;-0.79767047254336431727 0.26452348793500340962 0.44407166671429854077 0.39705924585419682948 1.0046752107429577361 -1.1983628006054949378 -0.24453756664923120834 -0.34218400322822328796 -0.43333338899283219137 -1.031529572932707417 -2.0302834119923329759 -0.11432256134027089789 -0.42037087137641243606 -0.635382206829227969 -2.0614765575583122725 1.2941247553001780091 -1.7726245605035149122 -0.90053308544909016486 0.75488354385453126927 1.2598766909210046272;-0.18991815656897079179 -0.043270523907141866804 -1.0481844324869056262 0.40610665233682169406 -0.0018749428394769108303 0.00026749384039947542852 -0.083817684048716839529 -0.00015346270543200148453 -0.0084219365520667686942 0.14203865129930229316 0.17476670929111240427 0.039099571572104765971 -0.062890611248467606642 -0.092103718703877776597 0.12776169535959325252 0.13080974284140362007 -0.13789562143137035211 -0.052991593093148998961 0.023714749103704808875 -0.064255405291172171256;0.097530150823504122015 -0.13196008601104144975 0.046618966411538731442 -0.51352027413214118567 -0.10807989846777273091 -0.19594579802204883934 0.22127778612673135172 0.12912277334687158326 -0.0014751855223808568382 0.10340146800815419592 0.031355446556398103297 0.16389181186970280812 0.080159603881544957971 -0.27645406476619366343 -0.34569998662189038718 0.080553349118493239311 -0.37375607263784116352 -0.040763655218558819682 0.032201624803224797311 0.15109403032038742909];

% Layer 3
b3 = [-0.60393603582894050952;-0.56153636467403744703];
LW3_2 = [0.40182137288461483582 0.12067400796221708359 -0.39552053701546791009 0.0027381823022213421653 -0.10566313991512286952 0.81266883395901157172 1.4439909484924264849 -0.87863645373822429896;0.23137094218513057364 0.084314758499023104155 -0.33971479871075194179 -0.094569524630851678482 -0.099097532855605380275 0.89836619322671840138 1.301884827059782479 -0.62480113706697948395];

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
