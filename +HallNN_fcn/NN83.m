function [y1] = NN83(x1)
%NN83 neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 22:52:30.
% 
% [y1] = NN83(x1) takes these arguments:
%   x = 13xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1.gain = [1.99954882377404;1.99840912461659;2;2;1.99808957651055;1.99849929994467;1.99813146949417;1.99816123745006;2;1.9989189734329;1.99830617265198;1.99922043490435;1.99815121069195];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.12255873527410920953;0.069880517047376328188;0.21578664605038280766;0.037452783535629433076;0.15889401953237330956;-0.0078922774371546315642;0.0035680517256632482762;1.0580434690893889549;-0.083357970221562230351;0.087509216275444542954;-0.36478389477959149989;0.94037223527321744942;-0.44685687062253526269;0.54737433041208127804;-0.22836122530946187381;-0.011819894216900351039;0.37527604702529449021;-0.21557255018110568767;1.72440583346779297;-0.26096549189927326085];
IW1_1 = [-0.04873109625743615031 -0.99881340653902239701 0.3911774783920421994 -0.2220813941752461107 0.096154375355777269019 -0.14613363288462383194 0.23444080146426860001 0.71772347560330007621 0.075209558172330243098 -0.72750114945741950478 -0.32371696183353115828 -0.034366016369711031853 0.0069220110679141695631;0.030754473225664491487 -1.1602263873730878085 0.35942358528413054231 -0.37444309229700944996 0.76453271557032520533 0.43869991804872054963 -0.94949218465862461613 0.21718150927955076623 -0.057400063621350493059 0.63747135157900758351 0.042406011054486617007 0.0059317177338639587303 0.071970981638885919707;-0.089570988385135205667 -0.38797118574055494689 -0.13139965946387710538 0.19237095305372908793 0.38316486370150404372 0.023644715718621225203 -0.26993823504685804382 0.22893812264876095552 -0.024935472670871292739 -0.010811536626768578073 0.0098338753483224357332 -0.0037817533969734889142 -0.028349803918863648172;-0.59797653376901760414 -0.27657000350225569019 0.21720790503877102084 0.36161024993180002074 -0.50706616621744060591 0.063206356732098525897 0.36662239839645410777 0.26925433201421467011 -0.14055060534308116971 0.32769977928306426307 -0.0024666685600286987479 -0.1287061754568043781 -0.067958229237180187154;0.28434145583668146706 -0.57577953723062136326 -0.38021078031008648246 0.4162602964473231526 0.082177686912522668328 0.29206864649370595233 0.249232839947552548 -0.2454596830350419534 -0.2037297224521231187 -0.19134330110626851518 -0.015911456112410925223 0.1207840633735827035 0.076831619488256314154;-0.42978066700647743481 -0.45299771808160926856 0.60496578766350239675 -0.21911961791094472507 0.67048583529177285811 -0.11060106257863386303 -0.3414811251978797646 -0.52627630531224589561 0.06659826603940820211 -0.7759601919958998284 -0.20900623590630970638 -0.01268071052341060434 -0.01301384413874920494;0.020504962345005726954 -0.22290468281817299445 -0.01549180820688756885 0.082520864854593020676 0.0028365971418301090187 -0.27857732333028850658 0.26135044869962698044 0.89206757541088088637 -0.081785238662768142137 -0.12546950744599905114 0.24796566302164183004 -0.0092817505726656562848 -0.073189165163837097139;-0.2851042563630374338 0.93050503420948338373 -0.042618987598933275041 0.42281731347721518421 -0.61786274131428819967 -0.1617604205708479681 0.21735813104593698641 -1.0539658220303800285 0.069456534124285040921 -0.18852524595178341005 -0.014621323082356595807 0.02525328395235716325 -0.027129403236572736946;-0.086793178002166168628 -0.0037283806249417093814 0.12398114346556181331 0.54631860452566083364 0.32964726883929756918 0.26458260385718335161 -0.21340858608024157461 -0.25170150896391885187 -0.54941339925278964706 0.045818395193123508113 0.024777035330925376716 -0.14653887006585267216 -0.48637595345221840137;0.26998275062955956471 -0.074168008359478426739 -0.16248938921570643679 0.24923277505436425283 0.079237247286071135766 -0.0012288860299835909114 -0.068481689709156121082 -0.044192938579779801611 -0.0085138585577469588772 -0.017064013121986007504 -0.038265304004097638879 -0.015885956342000056435 -0.015474013731962502766;0.25596338525671252784 0.12516758419359158561 -0.12301733686642214727 -0.64170788691657831837 0.38980347254966851711 0.79660225420345276426 -0.66178535163937779462 0.21250363038681449601 0.049275171745071753848 0.56980251139726456611 -0.44454684370195707288 0.35545071869444605506 -0.062757446848552250751;0.8637011620003730572 -0.10444502170385275286 -0.049354357184416212945 -0.055499716318800108861 0.22559807022252797992 -0.18215180904442887511 0.76802071982220776469 -0.91067814991545348313 0.37717975173107864961 -1.0810204033055994532 0.15788433831872558666 -0.062833474903568969916 0.12485667864608811228;-0.47586459559389171492 -0.68198423153047527112 -0.74599474477232496916 0.46419225177449385811 0.30457619085431092421 0.40563574057873308387 -0.25844833243405651579 -0.40370571478649730501 0.022423301751000480203 0.22174856031491715913 -0.092142868618996412455 -0.017820800598451255053 -0.20142836802709138122;-1.551173553843097741 0.52341600216757555586 0.96551533110934417703 0.2046163796920889566 -0.49131448118863030317 -0.20687536218285718093 0.8986178898953337546 -0.14312544747791405841 -0.058203279072517960513 -0.20057479876968015153 0.25170760795057350867 -0.064116957138196900567 -0.30593955782809545418;-0.18869959505954883583 -0.079306296565496731121 -0.25498352504203425362 0.05759206684835097706 -0.040656292953295897208 0.18344011879069671567 0.67092090768909173892 0.011642805626786602649 -0.27417564689650553156 -0.30939799387972771738 -0.11619258883465655163 0.030468134662774863652 -0.37431097118885453545;-0.093378334384118952261 -0.39661152946183969581 0.10169451737046658613 -0.2475728034301338254 0.14027002847292477372 0.41596597670140966763 -0.43327658378720140808 -1.762292045466883339 0.023782078831038586803 0.3708842703680755859 -0.057543259299601544343 0.031233184207121175019 0.081841528185194412082;0.72937721852619818996 -0.16215911720560610698 -0.58764496111169461923 -2.0184539662171094854 -0.60480888423784828767 -0.072076497301403086304 0.096332262899942830425 0.1050927276829997925 -0.37340462582729738727 0.51843938890897012239 -0.15701139634433219916 -0.16132347868331653462 -0.022393948652047569042;0.4277525501840225286 0.069935287781401692109 0.29438456189096029458 -0.74524506007245427952 0.54091168682931178857 0.67666774535946172531 -0.37305708967354650474 0.26379778454562863299 0.045963925700317438683 0.46492351834700623714 0.21367952676610382778 -0.074686813452123218537 0.042392952835096024689;1.9407147114368499263 0.75953947355775952577 -1.4896801636812033731 0.12755247278420986379 -0.25294616653271795315 -0.33280765579568433177 0.89027123251223772016 -0.84425941795268300005 0.11364691526710291036 -0.67965561312224187507 -0.14045633312819819083 0.036130966642147162304 0.011114444383592284038;0.25939693358250265343 -0.55479756852467410067 -0.43391475553162511813 -0.4900634352874139732 0.023003903000638471071 0.55824575494346495574 0.28806896990467234509 -0.36195489696329924412 0.33942712181985590369 -0.12022713966653307549 -0.29299146600830228149 -0.20608924880706608329 -0.2056059584270480689];

% Layer 2
b2 = [0.26780992191961877635;-0.024237940590275983826;-1.69357104085861776;0.33020627852917600054;-0.031919433561517177034;0.028860056582462829955;-0.15085657456866902182;0.22774029379982135834];
LW2_1 = [0.21608067332580929709 -0.12078300585064412342 0.20326235260905695101 0.11645652497691585947 0.038424089787622878189 0.3789940736051511383 -0.29929825826140116218 0.050997309879210556138 -0.046423802934152956501 -0.42730756611573522674 -0.059323944640480831547 -0.25477383206738046795 0.20226575216410552782 0.17647617142468888241 -0.2213618493964125411 -0.2462588886574001712 -0.052162066633578071939 0.26525827107690619355 -0.33783091207287258007 -0.14018454344609920792;-0.014066774665583169079 0.11063208266839248683 -0.24991310258109947218 0.031662904734766739168 -0.15701758994757400445 -0.19983238457376625852 0.061362094344359185971 0.1045860515965669757 -0.068917241635607431882 -0.14458384994811221413 -0.14309005941756283709 0.09285622846750167525 0.16191640730039161267 -0.076926298970602241534 0.22947657507229746177 -0.18569338548634392594 0.053466364778482058928 -0.29140895538834460377 0.17808524668314995587 0.022243189283008831664;1.3370295911275829059 -0.34125084018698348398 -0.4407270798403795653 0.54538416474715989413 -1.1993766082263535022 1.0378010746784862572 -0.57472508504214325775 -0.95290136906963063268 -0.26904029466619955313 0.2359174652076374834 0.79027931826198627441 0.82669425952389619106 -1.6416590020911274728 1.6739934070249362641 0.41380557491802560932 1.8576045008335013708 0.89028248472565973959 0.85954213431715587035 -2.5554510997713366649 -0.66392088352795997075;0.1063212259247225816 -0.17067630651009790821 0.19813377403837037494 -0.246432754824247785 -0.095099055875043123076 0.43113439908204698181 0.15337663957548941518 -0.61254614573050170101 0.29932320080870422618 -0.32016774583981766344 0.37245290689583548938 -0.03513227438464040786 -0.29432992676651686947 0.022417648153265405664 -0.41542976151449095834 0.75085591841211196673 -0.10000231587152010049 0.3176004100679616915 -0.61921767055254284262 -0.19302367333075531874;-0.028373761513373247506 0.0048075161370043049958 0.61394592651940727013 -0.046010374684385894895 -0.13593169118416975616 0.049389222224614975909 -0.23271195494511212609 0.23892791757330442137 0.029153908133040334905 -0.46912044312293466053 0.016882712402585337702 0.25084196366147409885 0.33144911980840530141 -0.16400227796699629046 0.057206026059150531793 -0.077695458777691697727 -0.029091370432559351261 -0.053412367217262543551 0.22255287743934082378 -0.16375843810366197406;0.16990528394553028324 0.1386908422802178853 0.086353113065668021808 0.096460961409246037679 0.29401754417341119385 0.079239272404098101688 -0.033717309189131634828 0.14267913819573641021 -0.032794714394235389976 0.20526864891983193084 0.086399793028931179406 0.13167262367191731598 -0.096336862833680189655 0.079560338489054249766 -0.069855102562117243892 0.14927591952365690253 -0.083600035457408405914 0.043534199368866884072 -0.1483213069069333756 0.053269137910147028314;0.01349240776621881896 -0.32588064028334484679 0.36559666021198289343 0.2923085388763056458 0.40629403022452581506 -0.39338374067745485174 0.075330982747299921054 0.0023411082416599972994 -0.0598000563274389621 -0.82427800376717097475 -0.0045699272222954889777 0.18280705273677894107 -0.34145502924093074082 0.048332503762471237807 0.071445812079996118271 0.17205314407714772384 -0.0094297407831367887482 -0.21708954638041641494 0.11433759837033161122 0.32065723488087927162;-0.16563202141697727643 0.65564066040090462906 0.34555393000854695007 -0.021415698966454628505 0.017488290644412460978 0.23371209356283131897 -0.32267326414238345356 0.22768600167288688407 0.01386461540560727547 -0.36752210492909298534 0.045864660292089340365 -0.078416132025306797826 -0.071028455793417086261 0.030598304877862015499 -0.01345560079807189835 -0.029432879118807885738 0.02208219003559085819 0.13868918598186399938 -0.017648070155800821973 -0.014703900881984493035];

% Layer 3
b3 = [-0.63313847803833001038;-0.57469935681265194027];
LW3_2 = [-0.81468764733395504507 -0.67755494463226684232 -1.1890428734396121246 -0.43612280874455933155 -0.68396597813610504613 0.59965043212949364904 -0.88527699034574203196 -0.73407793322478664955;-0.76480175878043765714 -0.13392706013408819943 -1.2045828676832399662 -0.23309704287042584681 -0.72547956705201654959 0.73066855822654641628 -0.8435455457160052628 -0.61960004909310040233];

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
