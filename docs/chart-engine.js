'use strict';
window.ProTEUSCharts=(()=>{const module={exports:{}};
// Component accents match the overview and its expanded video scenes.
// Published assay palettes and continuous numerical color scales stay separate.
const C={ink:'#2C3E50',muted:'#7F8C8D',blue:'#5290D3',structure:'#DA6457',teal:'#51AA8E',rose:'#DA6457',gold:'#E74C3C',paper:'#FFFFFF',grid:'#EAECEF',axis:'#9AA3A8'};
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
function text(c,s,x,y,size=36,col=C.ink,weight=400,align='left'){c.save();if(c.proteusCssUnit){const m=c.getTransform(),scale=Math.hypot(m.a,m.b)*c.proteusCssUnit,effective=size*scale;size=(effective<20?18:effective<32?24:40)/scale;}c.font=weight+' '+size+'px Georgia, "Times New Roman", serif';c.fillStyle=col;c.textAlign=align;c.textBaseline='middle';c.fillText(String(s),x,y);c.restore();}
const require=name=>{if(name!=='./overlay')throw Error('Unexpected dependency');return{C,text,clamp};};
(function(){
'use strict';
/* Native Canvas data graphics. No figure raster, SVG image, or external data fetch.
 * Numerical values are preserved from the original project-page chart dataset.
 * Animations reveal measured marks; their movement is not a simulation result.
 */
const {C,text,clamp}=require('./overlay');
/* DATA_START */
const DATA={"zero":[{"name":"ProTEUS","mean":0.2708966009335355,"sem":0.05342189930450411,"n":19},{"name":"ESM-IF1","mean":0.2579818735518448,"sem":0.04552624384056675,"n":19},{"name":"ESM3-1.4B","mean":0.2553539991367496,"sem":0.05373793038766104,"n":19},{"name":"MSA Transformer","mean":0.24337352829603262,"sem":0.05063111632193234,"n":19},{"name":"SaProt-650M","mean":0.23067843000754523,"sem":0.05554350694874763,"n":19},{"name":"ESM-2-3B","mean":0.22402423137160768,"sem":0.047168696543224385,"n":19},{"name":"ESM-2-650M","mean":0.212701053252733,"sem":0.05286298514526325,"n":19},{"name":"Tranception","mean":0.205008849997937,"sem":0.04903239083641996,"n":19},{"name":"ESM-C-600M","mean":0.19766608923387255,"sem":0.05240008518365164,"n":19},{"name":"ESM-C-300M","mean":0.19675501573002582,"sem":0.05303146338606109,"n":19},{"name":"MSA-Pairformer","mean":0.1956212182987164,"sem":0.04797354196804021,"n":19},{"name":"PoET-2","mean":0.18793820648966078,"sem":0.043922805218081167,"n":19},{"name":"ESM-1v","mean":0.17827962313536996,"sem":0.051349766861308675,"n":19},{"name":"AMPLIFY-120M","mean":0.13138111528179605,"sem":0.04578138233041406,"n":19},{"name":"AMPLIFY-350M","mean":0.12501639699580433,"sem":0.04400432505379175,"n":19}],"fine":[{"name":"ProTEUS","mean":0.5490182315789474,"sem":0.05365237713271151,"values":[{"dataset":"AAV","value":0.5302328000000001},{"dataset":"GB1","value":0.6628258},{"dataset":"GFP","value":0.8427070000000001},{"dataset":"TrpB","value":0.3626131999999999},{"dataset":"urn_mavedb_00000072-a-1","value":0.3810966},{"dataset":"urn_mavedb_00000077-a-1","value":0.5915845999999999},{"dataset":"urn_mavedb_00000077-b-1","value":0.7334636},{"dataset":"urn_mavedb_00000079-a-1","value":0.4258319999999999},{"dataset":"urn_mavedb_00000090-a-1","value":0.7399024},{"dataset":"urn_mavedb_00000091-a-1","value":0.0},{"dataset":"urn_mavedb_00000105-a-1","value":0.943176},{"dataset":"urn_mavedb_00000659-a-1","value":0.6244782},{"dataset":"urn_mavedb_00000665-a-1","value":0.5688422},{"dataset":"urn_mavedb_00001207-a-1","value":0.1647337999999999},{"dataset":"urn_mavedb_00001212-a-1","value":0.7093646},{"dataset":"urn_mavedb_00001217-a-3","value":0.5476561999999999},{"dataset":"urn_mavedb_00001227-a-1","value":0.7871775999999999},{"dataset":"urn_mavedb_00001239-a-1","value":0.4368401999999999},{"dataset":"urn_mavedb_00001243-a-1","value":0.3788196}]},{"name":"ProteinNPT","mean":0.36670332982730636,"sem":0.06124613665315524,"values":[{"dataset":"AAV","value":0.8697753111373541},{"dataset":"GB1","value":0.5442140772511479},{"dataset":"GFP","value":0.8015706093345385},{"dataset":"TrpB","value":0.2692895214480769},{"dataset":"urn_mavedb_00000072-a-1","value":0.3025},{"dataset":"urn_mavedb_00000077-a-1","value":0.117},{"dataset":"urn_mavedb_00000077-b-1","value":0.3031},{"dataset":"urn_mavedb_00000079-a-1","value":0.2276},{"dataset":"urn_mavedb_00000090-a-1","value":0.5016},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0413},{"dataset":"urn_mavedb_00000105-a-1","value":0.9341137475477046},{"dataset":"urn_mavedb_00000659-a-1","value":0.1859},{"dataset":"urn_mavedb_00000665-a-1","value":0.3368},{"dataset":"urn_mavedb_00001207-a-1","value":0.0526},{"dataset":"urn_mavedb_00001212-a-1","value":0.3943},{"dataset":"urn_mavedb_00001217-a-3","value":0.4443},{"dataset":"urn_mavedb_00001227-a-1","value":0.2673},{"dataset":"urn_mavedb_00001239-a-1","value":0.3178},{"dataset":"urn_mavedb_00001243-a-1","value":0.1389}]}],"selection":[{"name":"ProTEUS","x":5,"y":0.7384615384615386,"sem":0.08284868934053084},{"name":"MULTI-evolve","x":5,"y":0.6461538461538462,"sem":0.08815442788549274},{"name":"ProTEUS","x":10,"y":0.753846153846154,"sem":0.08519614218377515},{"name":"MULTI-evolve","x":10,"y":0.6461538461538462,"sem":0.08367778908759345},{"name":"ProTEUS","x":15,"y":0.7282051282051282,"sem":0.08753071531276657},{"name":"MULTI-evolve","x":15,"y":0.6615384615384616,"sem":0.06978254316906052},{"name":"ProTEUS","x":24,"y":0.7435897435897435,"sem":0.07321689623555304},{"name":"MULTI-evolve","x":24,"y":0.6794871794871794,"sem":0.06526719807393704},{"name":"ProTEUS","x":50,"y":0.723962703962704,"sem":0.07079090208737862},{"name":"MULTI-evolve","x":50,"y":0.6832167832167833,"sem":0.06322784419602147},{"name":"ProTEUS","x":100,"y":0.6947133990507484,"sem":0.07009868037483122},{"name":"MULTI-evolve","x":100,"y":0.6578136321509815,"sem":0.06225262979139748}],"single":[{"name":"ProTEUS","x":1,"y":0.3084375957964232,"sem":0.0911105695211213},{"name":"ProTEUS","x":2,"y":0.5628413899615614,"sem":0.081804329979377},{"name":"ProTEUS","x":3,"y":0.6120103362306846,"sem":0.079805589578009},{"name":"ProTEUS","x":4,"y":0.6987885348226411,"sem":0.0922656363337565},{"name":"ProTEUS","x":5,"y":0.7407734004293498,"sem":0.0864594744243732},{"name":"EVOLVEpro","x":1,"y":0.2689863484012691,"sem":0.0436558258477791},{"name":"EVOLVEpro","x":2,"y":0.3728975817911066,"sem":0.0553331903678499},{"name":"EVOLVEpro","x":3,"y":0.453407457157699,"sem":0.0637345210930066},{"name":"EVOLVEpro","x":4,"y":0.5019851590100431,"sem":0.0627712990717716},{"name":"EVOLVEpro","x":5,"y":0.5550121223461167,"sem":0.0665131428443342}],"quality":[{"name":"AlphaFold3","metric":"Complex LDDT","mean":0.65,"error":0.06},{"name":"AlphaFold3","metric":"Prot-Prot iLDDT","mean":0.83,"error":0.05},{"name":"AlphaFold3","metric":"TM-score","mean":0.39,"error":0.08},{"name":"AlphaFold3","metric":"Physical Validity","mean":0.74,"error":0.11},{"name":"ProTEUS","metric":"Complex LDDT","mean":0.63,"error":0.06},{"name":"ProTEUS","metric":"Prot-Prot iLDDT","mean":0.41,"error":0.15},{"name":"ProTEUS","metric":"TM-score","mean":0.36,"error":0.07},{"name":"ProTEUS","metric":"Physical Validity","mean":1.0,"error":0.0}],"runtime":[{"name":"ProTEUS","bin":"0-1000","seconds":1.96},{"name":"ProTEUS","bin":"1000-2000","seconds":2.16},{"name":"ProTEUS","bin":"2000-3000","seconds":2.41},{"name":"ProTEUS","bin":"3000-4000","seconds":3.07},{"name":"ProTEUS","bin":"4000+","seconds":10.86},{"name":"AlphaFold3","bin":"0-1000","seconds":15.3},{"name":"AlphaFold3","bin":"1000-2000","seconds":18.2},{"name":"AlphaFold3","bin":"2000-3000","seconds":25.1},{"name":"AlphaFold3","bin":"3000-4000","seconds":33.2},{"name":"AlphaFold3","bin":"4000+","seconds":90.1}],"higher":[{"protein":"CR9114","phenotype":"binding","k":1,"ours":0.2707723255300303,"baseline":-0.196408105006707},{"protein":"CreiLOV","phenotype":"fluorescence","k":1,"ours":0.1444619020072865,"baseline":0.4506413297345697},{"protein":"GB1","phenotype":"binding","k":1,"ours":0.7689723327983158,"baseline":0.0247824865376643},{"protein":"HIV-1","phenotype":"activity","k":1,"ours":0.1451758047419447,"baseline":-0.0148363022671469},{"protein":"RhlA","phenotype":"activity_%Rha-(C8-C10)","k":1,"ours":0.3899943377903924,"baseline":0.2979663383807346},{"protein":"RhlA","phenotype":"activity_RLs","k":1,"ours":0.154517644220363,"baseline":0.1036560590765617},{"protein":"RhlA","phenotype":"activity_Rha-(C8-C10)","k":1,"ours":0.3048074571354569,"baseline":0.2367854185794232},{"protein":"SaCas9","phenotype":"activity","k":1,"ours":0.2550697554505062,"baseline":-0.1594044010775455},{"protein":"aav","phenotype":"fitness","k":1,"ours":0.130098528007233,"baseline":0.2723816592015567},{"protein":"eqFP611","phenotype":"fluorescence_blue","k":1,"ours":0.5349797945412421,"baseline":0.5994411374885226},{"protein":"eqFP611","phenotype":"fluorescence_combined","k":1,"ours":0.456968855128712,"baseline":0.1186329892437673},{"protein":"eqFP611","phenotype":"fluorescence_red","k":1,"ours":0.3421445498286685,"baseline":-0.013595843182425},{"protein":"trpb","phenotype":"activity","k":1,"ours":0.4830197260741357,"baseline":0.1447978977446477},{"protein":"CR6261","phenotype":"binding_H9","k":2,"ours":0.6736248997245935,"baseline":0.4223497476384715},{"protein":"CR9114","phenotype":"binding","k":2,"ours":0.3274165245385518,"baseline":0.54833219052182},{"protein":"CreiLOV","phenotype":"fluorescence","k":2,"ours":0.3834749936356725,"baseline":0.6579134924993004},{"protein":"GB1","phenotype":"binding","k":2,"ours":0.7411897675325421,"baseline":0.3733958986574311},{"protein":"HIV-1","phenotype":"activity","k":2,"ours":0.3043204389102473,"baseline":0.0372285352524549},{"protein":"RhlA","phenotype":"activity_%Rha-(C8-C10)","k":2,"ours":0.1892198820028709,"baseline":0.0785756216994679},{"protein":"RhlA","phenotype":"activity_RLs","k":2,"ours":0.4275038908359897,"baseline":0.3424935579309504},{"protein":"RhlA","phenotype":"activity_Rha-(C8-C10)","k":2,"ours":0.3370413786509802,"baseline":0.133682262028111},{"protein":"SaCas9","phenotype":"activity","k":2,"ours":0.2449493884904399,"baseline":-0.0910352800133058},{"protein":"aav","phenotype":"fitness","k":2,"ours":0.4607903016478843,"baseline":0.3714331134688126},{"protein":"eqFP611","phenotype":"fluorescence_blue","k":2,"ours":0.7522735105590459,"baseline":0.5728945343815718},{"protein":"eqFP611","phenotype":"fluorescence_combined","k":2,"ours":0.4633278262555638,"baseline":0.1998355724361528},{"protein":"eqFP611","phenotype":"fluorescence_red","k":2,"ours":0.3061251996010795,"baseline":0.2051037462209508},{"protein":"gfp","phenotype":"fluorescence","k":2,"ours":0.6465530230674982,"baseline":0.1781912807701087},{"protein":"trpb","phenotype":"activity","k":2,"ours":0.427393831572,"baseline":0.2198571478733432},{"protein":"CR6261","phenotype":"binding_H9","k":3,"ours":0.9061636978240089,"baseline":0.7287984940660766},{"protein":"CR9114","phenotype":"binding","k":3,"ours":0.5324984840867807,"baseline":0.665988935030457},{"protein":"CreiLOV","phenotype":"fluorescence","k":3,"ours":0.7978255757676298,"baseline":0.8233613565152784},{"protein":"GB1","phenotype":"binding","k":3,"ours":0.5602759064413952,"baseline":0.3405801928062982},{"protein":"HIV-1","phenotype":"activity","k":3,"ours":0.303481650566991,"baseline":0.1099959774030353},{"protein":"SaCas9","phenotype":"activity","k":3,"ours":0.2653590872707105,"baseline":-0.0350632873960049},{"protein":"aav","phenotype":"fitness","k":3,"ours":0.4247545188676078,"baseline":0.6379677987466229},{"protein":"eqFP611","phenotype":"fluorescence_blue","k":3,"ours":0.7730342625407275,"baseline":0.678190006700016},{"protein":"eqFP611","phenotype":"fluorescence_combined","k":3,"ours":0.5976096864460322,"baseline":0.2299779325422745},{"protein":"eqFP611","phenotype":"fluorescence_red","k":3,"ours":0.6138008832149864,"baseline":0.5612174572122289},{"protein":"gfp","phenotype":"fluorescence","k":3,"ours":0.8042006229479453,"baseline":0.6918915774203985},{"protein":"trpb","phenotype":"activity","k":3,"ours":0.2740510273422623,"baseline":0.2422250013914247}],"multi":[{"name":"ProTEUS","x":1,"y":0.8917496505819879,"sem":0.03398110509596147},{"name":"ProTEUS","x":2,"y":0.9161609181186227,"sem":0.024298093823864417},{"name":"ProTEUS","x":3,"y":0.940962544743857,"sem":0.016571048429499306},{"name":"ProTEUS","x":4,"y":0.9550570300610896,"sem":0.014323941877149214},{"name":"ProTEUS","x":5,"y":0.9575894890004949,"sem":0.01441728827002935},{"name":"EVOLVEpro","x":1,"y":0.8151573006187935,"sem":0.0522047801205236},{"name":"EVOLVEpro","x":2,"y":0.868419568563517,"sem":0.03986730406176138},{"name":"EVOLVEpro","x":3,"y":0.898350086186957,"sem":0.031331226838849056},{"name":"EVOLVEpro","x":4,"y":0.921869995213294,"sem":0.02141617059094339},{"name":"EVOLVEpro","x":5,"y":0.9405283538111012,"sem":0.015901528424335835},{"name":"Additive","x":1,"y":0.793817057753812,"sem":0.057638409886548204},{"name":"Additive","x":2,"y":0.848735254206582,"sem":0.0431092216570901},{"name":"Additive","x":3,"y":0.8909666026357879,"sem":0.030273187115868846},{"name":"Additive","x":4,"y":0.9134753061348247,"sem":0.025557451767304618},{"name":"Additive","x":5,"y":0.9282242777660649,"sem":0.020618524109519094},{"name":"MULTI-evolve","x":1,"y":0.7632902002807513,"sem":0.06349876646196743},{"name":"MULTI-evolve","x":2,"y":0.7911030395453872,"sem":0.05960015214853421},{"name":"MULTI-evolve","x":3,"y":0.8220351324322116,"sem":0.05060276491379169},{"name":"MULTI-evolve","x":4,"y":0.8502510331534753,"sem":0.041456860876021455},{"name":"MULTI-evolve","x":5,"y":0.8781534869839892,"sem":0.03362438113427511}],"scaling":{"fractions":[1.25,2.5,5,10,20,40,80,100],"source_x":[457.5198,479.5062,501.4925,523.4788,545.4651,567.4515,589.4378,596.5158],"source_mse_y":[61.4151,71.20679,80.26099,92.83362,98.4314,106.285587,112.08728,115.31061],"source_rho_y":[202.99469,187.45679,179.1883,167.83313,162.09558,156.24329,153.30078,151.52149],"source_bounds":{"x":[453.465,600.949],"mse_y":[56,121],"rho_y":[145,210]},"mse_ticks":[[0.27,59.51947],[0.18,85.913578],[0.12,112.30768]],"rho_ticks":[[0.84,204.34778],[0.89,176.08698],[0.94,147.82599]],"fit_source_endpoints":[[457.5198,63.47638],[596.5158,116.70038]],"reported_mse_endpoints":[0.262,0.115],"reported_rho_endpoints":[0.842,0.933],"fit_exponent":-0.187,"fit_r_squared":0.989,"recovery":"Eight measured marker centers recovered from source SVG; no raster trace, interpolation, or invented observations."}};
/* DATA_END */
const ZERO_ASSAYS={"assays":[{"id":"AAV","name":"AAV","label":"AAV","short_label":"AAV","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"GB1","name":"GB1","label":"GB1","short_label":"GB1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"GFP","name":"GFP","label":"GFP","short_label":"GFP","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"TrpB","name":"TrpB","label":"TrpB","short_label":"TrpB","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000072-a-1","name":"urn_mavedb_00000072-a-1","label":"MaveDB 72-a-1","short_label":"72-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000077-a-1","name":"urn_mavedb_00000077-a-1","label":"MaveDB 77-a-1","short_label":"77-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000077-b-1","name":"urn_mavedb_00000077-b-1","label":"MaveDB 77-b-1","short_label":"77-b-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000079-a-1","name":"urn_mavedb_00000079-a-1","label":"MaveDB 79-a-1","short_label":"79-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000090-a-1","name":"urn_mavedb_00000090-a-1","label":"MaveDB 90-a-1","short_label":"90-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000091-a-1","name":"urn_mavedb_00000091-a-1","label":"MaveDB 91-a-1","short_label":"91-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000105-a-1","name":"urn_mavedb_00000105-a-1","label":"MaveDB 105-a-1","short_label":"105-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000659-a-1","name":"urn_mavedb_00000659-a-1","label":"MaveDB 659-a-1","short_label":"659-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00000665-a-1","name":"urn_mavedb_00000665-a-1","label":"MaveDB 665-a-1","short_label":"665-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001207-a-1","name":"urn_mavedb_00001207-a-1","label":"MaveDB 1207-a-1","short_label":"1207-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001212-a-1","name":"urn_mavedb_00001212-a-1","label":"MaveDB 1212-a-1","short_label":"1212-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001217-a-3","name":"urn_mavedb_00001217-a-3","label":"MaveDB 1217-a-3","short_label":"1217-a-3","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001227-a-1","name":"urn_mavedb_00001227-a-1","label":"MaveDB 1227-a-1","short_label":"1227-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001239-a-1","name":"urn_mavedb_00001239-a-1","label":"MaveDB 1239-a-1","short_label":"1239-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."},{"id":"urn_mavedb_00001243-a-1","name":"urn_mavedb_00001243-a-1","label":"MaveDB 1243-a-1","short_label":"1243-a-1","name_note":"Original source assay identifier; no unverified protein-name expansion."}],"models":[{"id":"AMPLIFY-350M","source_key":"AMPLIFY-350M","label":"AMPLIFY-350M","short_label":"AMPLIFY-350M","mean":0.1250163969958043,"sem":0.04400432505379175,"source_mean":0.12501639699580433,"source_sem":0.04400432505379175,"n_assays":19,"n_observed":18,"n_missing":1,"mean_observed_only":0.1319617523844601,"summary_max_abs_error":2.7755575615628914e-17,"name":"AMPLIFY-350M","values":[{"dataset":"AAV","value":null},{"dataset":"GB1","value":0.0005401969237792},{"dataset":"GFP","value":0.0047878311042452},{"dataset":"TrpB","value":0.0045113823444614},{"dataset":"urn_mavedb_00000072-a-1","value":0.3037684436030111},{"dataset":"urn_mavedb_00000077-a-1","value":-0.120490026852867},{"dataset":"urn_mavedb_00000077-b-1","value":-0.1013116387869588},{"dataset":"urn_mavedb_00000079-a-1","value":0.2104466139756685},{"dataset":"urn_mavedb_00000090-a-1","value":0.4516004159783394},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0393372824879674},{"dataset":"urn_mavedb_00000105-a-1","value":0.2456822204541147},{"dataset":"urn_mavedb_00000659-a-1","value":0.2149060585784472},{"dataset":"urn_mavedb_00000665-a-1","value":0.277427154650995},{"dataset":"urn_mavedb_00001207-a-1","value":0.0884510702081339},{"dataset":"urn_mavedb_00001212-a-1","value":0.2100140404065593},{"dataset":"urn_mavedb_00001217-a-3","value":0.4338654131247966},{"dataset":"urn_mavedb_00001227-a-1","value":0.2225539825920628},{"dataset":"urn_mavedb_00001239-a-1","value":0.227455226555946},{"dataset":"urn_mavedb_00001243-a-1","value":-0.2595595594524852}]},{"id":"AMPLIFY-120M","source_key":"AMPLIFY-120M","label":"AMPLIFY-120M","short_label":"AMPLIFY-120M","mean":0.13138111528179605,"sem":0.04578138233041406,"source_mean":0.13138111528179605,"source_sem":0.04578138233041406,"n_assays":19,"n_observed":18,"n_missing":1,"mean_observed_only":0.1386800661307847,"summary_max_abs_error":0.0,"name":"AMPLIFY-120M","values":[{"dataset":"AAV","value":null},{"dataset":"GB1","value":0.0016643579455718},{"dataset":"GFP","value":0.004454206978867},{"dataset":"TrpB","value":0.0023884534558633},{"dataset":"urn_mavedb_00000072-a-1","value":0.247189051765793},{"dataset":"urn_mavedb_00000077-a-1","value":-0.1437316105269253},{"dataset":"urn_mavedb_00000077-b-1","value":-0.0949232509215773},{"dataset":"urn_mavedb_00000079-a-1","value":0.2265686332880601},{"dataset":"urn_mavedb_00000090-a-1","value":0.5208904102722943},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0003393804763667},{"dataset":"urn_mavedb_00000105-a-1","value":0.2247324617095756},{"dataset":"urn_mavedb_00000659-a-1","value":0.2427491593015161},{"dataset":"urn_mavedb_00000665-a-1","value":0.2564622049816114},{"dataset":"urn_mavedb_00001207-a-1","value":0.1055439334576797},{"dataset":"urn_mavedb_00001212-a-1","value":0.2013661279601135},{"dataset":"urn_mavedb_00001217-a-3","value":0.4388675705901867},{"dataset":"urn_mavedb_00001227-a-1","value":0.3012184990477231},{"dataset":"urn_mavedb_00001239-a-1","value":0.2258491941490406},{"dataset":"urn_mavedb_00001243-a-1","value":-0.2647088326249021}]},{"id":"ESM-1v","source_key":"ESM-1v","label":"ESM-1v","short_label":"ESM-1v","mean":0.17827962313537,"sem":0.051349766861308675,"source_mean":0.17827962313536996,"source_sem":0.051349766861308675,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.17827962313537,"summary_max_abs_error":2.7755575615628914e-17,"name":"ESM-1v","values":[{"dataset":"AAV","value":0.2036232562901741},{"dataset":"GB1","value":0.1053183120831981},{"dataset":"GFP","value":0.1038076756602961},{"dataset":"TrpB","value":0.1784058531687422},{"dataset":"urn_mavedb_00000072-a-1","value":0.3814889967126284},{"dataset":"urn_mavedb_00000077-a-1","value":-0.1352161009886817},{"dataset":"urn_mavedb_00000077-b-1","value":-0.090867255018897},{"dataset":"urn_mavedb_00000079-a-1","value":0.2190162834125256},{"dataset":"urn_mavedb_00000090-a-1","value":0.582814001186581},{"dataset":"urn_mavedb_00000091-a-1","value":-0.1017215845982969},{"dataset":"urn_mavedb_00000105-a-1","value":0.2035308434700092},{"dataset":"urn_mavedb_00000659-a-1","value":0.1414038963535007},{"dataset":"urn_mavedb_00000665-a-1","value":0.4636924897203432},{"dataset":"urn_mavedb_00001207-a-1","value":0.1055032562452433},{"dataset":"urn_mavedb_00001212-a-1","value":0.226064529382802},{"dataset":"urn_mavedb_00001217-a-3","value":0.4250999547517848},{"dataset":"urn_mavedb_00001227-a-1","value":0.3518129465578339},{"dataset":"urn_mavedb_00001239-a-1","value":0.3256365397462127},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3021010545639701}]},{"id":"PoET-2","source_key":"PoET-2","label":"PoET-2","short_label":"PoET-2","mean":0.18793820648966078,"sem":0.04392280521808117,"source_mean":0.18793820648966078,"source_sem":0.043922805218081167,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.18793820648966078,"summary_max_abs_error":6.938893903907228e-18,"name":"PoET-2","values":[{"dataset":"AAV","value":-0.0367111933607623},{"dataset":"GB1","value":-0.0014100060761517},{"dataset":"GFP","value":0.0083575628812031},{"dataset":"TrpB","value":-0.0030532956938627},{"dataset":"urn_mavedb_00000072-a-1","value":0.4353516710237547},{"dataset":"urn_mavedb_00000077-a-1","value":0.1063426456096709},{"dataset":"urn_mavedb_00000077-b-1","value":0.2561104686766484},{"dataset":"urn_mavedb_00000079-a-1","value":0.1940053467496184},{"dataset":"urn_mavedb_00000090-a-1","value":0.4988071414766387},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0103357578658963},{"dataset":"urn_mavedb_00000105-a-1","value":0.3596447307826865},{"dataset":"urn_mavedb_00000659-a-1","value":0.2584279368334656},{"dataset":"urn_mavedb_00000665-a-1","value":0.3523585706221896},{"dataset":"urn_mavedb_00001207-a-1","value":0.0886125106978076},{"dataset":"urn_mavedb_00001212-a-1","value":0.2784619504973},{"dataset":"urn_mavedb_00001217-a-3","value":0.4495515795587701},{"dataset":"urn_mavedb_00001227-a-1","value":0.2648644032740588},{"dataset":"urn_mavedb_00001239-a-1","value":0.254797374678244},{"dataset":"urn_mavedb_00001243-a-1","value":-0.1833577170618285}]},{"id":"MSA-Pairformer","source_key":"MSA-Pairformer","label":"MSA-Pairformer","short_label":"MSA-Pairformer","mean":0.19562121829871637,"sem":0.04797354196804022,"source_mean":0.1956212182987164,"source_sem":0.04797354196804021,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.19562121829871637,"summary_max_abs_error":2.7755575615628914e-17,"name":"MSA-Pairformer","values":[{"dataset":"AAV","value":-0.0329365714625985},{"dataset":"GB1","value":-0.0003974186303411},{"dataset":"GFP","value":-0.0026906107535851},{"dataset":"TrpB","value":0.0029096272730719},{"dataset":"urn_mavedb_00000072-a-1","value":0.3060138231366766},{"dataset":"urn_mavedb_00000077-a-1","value":0.2149600557736399},{"dataset":"urn_mavedb_00000077-b-1","value":0.3217367357914525},{"dataset":"urn_mavedb_00000079-a-1","value":0.2550392519240108},{"dataset":"urn_mavedb_00000090-a-1","value":0.4338384788367656},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0438417869924719},{"dataset":"urn_mavedb_00000105-a-1","value":0.3951525502706561},{"dataset":"urn_mavedb_00000659-a-1","value":0.2633575417247977},{"dataset":"urn_mavedb_00000665-a-1","value":0.3446736609022875},{"dataset":"urn_mavedb_00001207-a-1","value":0.0993772719288293},{"dataset":"urn_mavedb_00001212-a-1","value":0.4408987752657808},{"dataset":"urn_mavedb_00001217-a-3","value":0.4241439851564537},{"dataset":"urn_mavedb_00001227-a-1","value":0.1786097183177723},{"dataset":"urn_mavedb_00001239-a-1","value":0.4097188216757789},{"dataset":"urn_mavedb_00001243-a-1","value":-0.2937607624633657}]},{"id":"ESM-C-300M","source_key":"ESM-C-300M","label":"ESM-C-300M","short_label":"ESM-C-300M","mean":0.19675501573002582,"sem":0.05303146338606109,"source_mean":0.19675501573002582,"source_sem":0.05303146338606109,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.19675501573002582,"summary_max_abs_error":0.0,"name":"ESM-C-300M","values":[{"dataset":"AAV","value":0.2083112247148055},{"dataset":"GB1","value":0.1247522482580979},{"dataset":"GFP","value":0.1141880822735474},{"dataset":"TrpB","value":0.1826077457253772},{"dataset":"urn_mavedb_00000072-a-1","value":0.3640750515028623},{"dataset":"urn_mavedb_00000077-a-1","value":-0.0925974595964356},{"dataset":"urn_mavedb_00000077-b-1","value":-0.087928585241808},{"dataset":"urn_mavedb_00000079-a-1","value":0.2375320305774191},{"dataset":"urn_mavedb_00000090-a-1","value":0.5987397790321286},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0913170327548356},{"dataset":"urn_mavedb_00000105-a-1","value":0.2990906323982873},{"dataset":"urn_mavedb_00000659-a-1","value":0.1865083993721573},{"dataset":"urn_mavedb_00000665-a-1","value":0.4787555177723143},{"dataset":"urn_mavedb_00001207-a-1","value":0.1074826449139416},{"dataset":"urn_mavedb_00001212-a-1","value":0.2482559295154373},{"dataset":"urn_mavedb_00001217-a-3","value":0.4592744126977001},{"dataset":"urn_mavedb_00001227-a-1","value":0.3808512122587527},{"dataset":"urn_mavedb_00001239-a-1","value":0.3541948843662167},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3344314189154752}]},{"id":"ESM-C-600M","source_key":"ESM-C-600M","label":"ESM-C-600M","short_label":"ESM-C-600M","mean":0.19766608923387255,"sem":0.052400085183651635,"source_mean":0.19766608923387255,"source_sem":0.05240008518365164,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.19766608923387255,"summary_max_abs_error":6.938893903907228e-18,"name":"ESM-C-600M","values":[{"dataset":"AAV","value":0.1796326059010684},{"dataset":"GB1","value":0.132756475138774},{"dataset":"GFP","value":0.1205658614992903},{"dataset":"TrpB","value":0.1841495178826501},{"dataset":"urn_mavedb_00000072-a-1","value":0.3912003222628514},{"dataset":"urn_mavedb_00000077-a-1","value":-0.0696896442273413},{"dataset":"urn_mavedb_00000077-b-1","value":-0.0392815493676423},{"dataset":"urn_mavedb_00000079-a-1","value":0.2433903430517648},{"dataset":"urn_mavedb_00000090-a-1","value":0.5494043094898767},{"dataset":"urn_mavedb_00000091-a-1","value":-0.1932082402399587},{"dataset":"urn_mavedb_00000105-a-1","value":0.2988724126771523},{"dataset":"urn_mavedb_00000659-a-1","value":0.2659420712057432},{"dataset":"urn_mavedb_00000665-a-1","value":0.4834627330592079},{"dataset":"urn_mavedb_00001207-a-1","value":0.1065200522640023},{"dataset":"urn_mavedb_00001212-a-1","value":0.2666680723912892},{"dataset":"urn_mavedb_00001217-a-3","value":0.4406666661469448},{"dataset":"urn_mavedb_00001227-a-1","value":0.3428095476169915},{"dataset":"urn_mavedb_00001239-a-1","value":0.3693627511103558},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3175686124194422}]},{"id":"Tranception","source_key":"Tranception","label":"Tranception","short_label":"Tranception","mean":0.205008849997937,"sem":0.04903239083641996,"source_mean":0.205008849997937,"source_sem":0.04903239083641996,"n_assays":19,"n_observed":17,"n_missing":2,"mean_observed_only":0.2291275382329884,"summary_max_abs_error":0.0,"name":"Tranception","values":[{"dataset":"AAV","value":0.502677375088636},{"dataset":"GB1","value":0.0446498875192626},{"dataset":"GFP","value":0.6434063171610751},{"dataset":"TrpB","value":null},{"dataset":"urn_mavedb_00000072-a-1","value":0.297336223732646},{"dataset":"urn_mavedb_00000077-a-1","value":-0.0421402799209588},{"dataset":"urn_mavedb_00000077-b-1","value":-0.0823936609268124},{"dataset":"urn_mavedb_00000079-a-1","value":0.1955868373680132},{"dataset":"urn_mavedb_00000090-a-1","value":0.4175464300779108},{"dataset":"urn_mavedb_00000091-a-1","value":-0.1417067752684191},{"dataset":"urn_mavedb_00000105-a-1","value":0.1704695747278154},{"dataset":"urn_mavedb_00000659-a-1","value":0.2233828622173266},{"dataset":"urn_mavedb_00000665-a-1","value":0.3995499222289161},{"dataset":"urn_mavedb_00001207-a-1","value":0.0985132607529573},{"dataset":"urn_mavedb_00001212-a-1","value":0.1520220889660057},{"dataset":"urn_mavedb_00001217-a-3","value":0.4056322292029561},{"dataset":"urn_mavedb_00001227-a-1","value":0.2990889818348247},{"dataset":"urn_mavedb_00001239-a-1","value":0.3115468751986476},{"dataset":"urn_mavedb_00001243-a-1","value":null}]},{"id":"ESM-2-650M","source_key":"ESM-2-650M","label":"ESM-2-650M","short_label":"ESM-2-650M","mean":0.212701053252733,"sem":0.05286298514526325,"source_mean":0.212701053252733,"source_sem":0.05286298514526325,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.212701053252733,"summary_max_abs_error":0.0,"name":"ESM-2-650M","values":[{"dataset":"AAV","value":0.2817639574017547},{"dataset":"GB1","value":0.125671080743577},{"dataset":"GFP","value":0.0911647607706747},{"dataset":"TrpB","value":0.1835102141068178},{"dataset":"urn_mavedb_00000072-a-1","value":0.3725369563299365},{"dataset":"urn_mavedb_00000077-a-1","value":-0.0867644324707393},{"dataset":"urn_mavedb_00000077-b-1","value":0.0053388099546666},{"dataset":"urn_mavedb_00000079-a-1","value":0.2213445619003434},{"dataset":"urn_mavedb_00000090-a-1","value":0.5516418456079397},{"dataset":"urn_mavedb_00000091-a-1","value":-0.116068122917438},{"dataset":"urn_mavedb_00000105-a-1","value":0.2797409009101526},{"dataset":"urn_mavedb_00000659-a-1","value":0.2576870797431033},{"dataset":"urn_mavedb_00000665-a-1","value":0.5080663727366953},{"dataset":"urn_mavedb_00001207-a-1","value":0.1068098246924519},{"dataset":"urn_mavedb_00001212-a-1","value":0.2754855484141019},{"dataset":"urn_mavedb_00001217-a-3","value":0.4841761207960656},{"dataset":"urn_mavedb_00001227-a-1","value":0.4404720564582377},{"dataset":"urn_mavedb_00001239-a-1","value":0.379205314176798},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3204628375532126}]},{"id":"ESM-2-3B","source_key":"ESM-2-3B","label":"ESM-2-3B","short_label":"ESM-2-3B","mean":0.22402423137160768,"sem":0.047168696543224385,"source_mean":0.22402423137160768,"source_sem":0.047168696543224385,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.22402423137160768,"summary_max_abs_error":0.0,"name":"ESM-2-3B","values":[{"dataset":"AAV","value":0.1915933650895485},{"dataset":"GB1","value":0.0671760693919187},{"dataset":"GFP","value":0.1472590100875566},{"dataset":"TrpB","value":0.1847531093524695},{"dataset":"urn_mavedb_00000072-a-1","value":0.3182369216145905},{"dataset":"urn_mavedb_00000077-a-1","value":0.0936034951940116},{"dataset":"urn_mavedb_00000077-b-1","value":0.2422140769064437},{"dataset":"urn_mavedb_00000079-a-1","value":0.2176981116672276},{"dataset":"urn_mavedb_00000090-a-1","value":0.5140681862977322},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0539306429717388},{"dataset":"urn_mavedb_00000105-a-1","value":0.2304852883446503},{"dataset":"urn_mavedb_00000659-a-1","value":0.2192441270816049},{"dataset":"urn_mavedb_00000665-a-1","value":0.5615753092040429},{"dataset":"urn_mavedb_00001207-a-1","value":0.1047064505528984},{"dataset":"urn_mavedb_00001212-a-1","value":0.3016502920806687},{"dataset":"urn_mavedb_00001217-a-3","value":0.4724848289050499},{"dataset":"urn_mavedb_00001227-a-1","value":0.3752085471556164},{"dataset":"urn_mavedb_00001239-a-1","value":0.3872544645225976},{"dataset":"urn_mavedb_00001243-a-1","value":-0.318820614416343}]},{"id":"SaProt-650M","source_key":"SaProt-650M","label":"SaProt-650M","short_label":"SaProt-650M","mean":0.23067843000754523,"sem":0.05554350694874763,"source_mean":0.23067843000754523,"source_sem":0.05554350694874763,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.23067843000754523,"summary_max_abs_error":0.0,"name":"SaProt-650M","values":[{"dataset":"AAV","value":0.2255568455754953},{"dataset":"GB1","value":0.1516095471010531},{"dataset":"GFP","value":0.6096103515200229},{"dataset":"TrpB","value":0.1088288782094439},{"dataset":"urn_mavedb_00000072-a-1","value":0.3837904531158409},{"dataset":"urn_mavedb_00000077-a-1","value":-0.076413674393506},{"dataset":"urn_mavedb_00000077-b-1","value":0.0089606076075336},{"dataset":"urn_mavedb_00000079-a-1","value":0.2310284917910126},{"dataset":"urn_mavedb_00000090-a-1","value":0.5044995306139901},{"dataset":"urn_mavedb_00000091-a-1","value":-0.136986301369863},{"dataset":"urn_mavedb_00000105-a-1","value":0.3564428238282577},{"dataset":"urn_mavedb_00000659-a-1","value":0.2365991024130791},{"dataset":"urn_mavedb_00000665-a-1","value":0.4902982909899319},{"dataset":"urn_mavedb_00001207-a-1","value":0.1122234256424461},{"dataset":"urn_mavedb_00001212-a-1","value":0.2852566245313391},{"dataset":"urn_mavedb_00001217-a-3","value":0.4712626119575014},{"dataset":"urn_mavedb_00001227-a-1","value":0.3518224702472217},{"dataset":"urn_mavedb_00001239-a-1","value":0.3971070303850757},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3286069396225167}]},{"id":"MSA Transformer","source_key":"MSA Transformer","label":"MSA Transformer","short_label":"MSA Transformer","mean":0.24337352829603262,"sem":0.05063111632193234,"source_mean":0.24337352829603262,"source_sem":0.05063111632193234,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.24337352829603262,"summary_max_abs_error":0.0,"name":"MSA Transformer","values":[{"dataset":"AAV","value":0.4730983363796934},{"dataset":"GB1","value":0.1049325679900447},{"dataset":"GFP","value":0.6580609799088106},{"dataset":"TrpB","value":0.1788178364770343},{"dataset":"urn_mavedb_00000072-a-1","value":0.331080748084658},{"dataset":"urn_mavedb_00000077-a-1","value":0.1073793602124282},{"dataset":"urn_mavedb_00000077-b-1","value":0.2839791835403151},{"dataset":"urn_mavedb_00000079-a-1","value":0.2252094710477341},{"dataset":"urn_mavedb_00000090-a-1","value":0.4941309575607158},{"dataset":"urn_mavedb_00000091-a-1","value":-0.1621621621621622},{"dataset":"urn_mavedb_00000105-a-1","value":0.1395620739102717},{"dataset":"urn_mavedb_00000659-a-1","value":0.2124199228023499},{"dataset":"urn_mavedb_00000665-a-1","value":0.3293476293192891},{"dataset":"urn_mavedb_00001207-a-1","value":0.0946058092896727},{"dataset":"urn_mavedb_00001212-a-1","value":0.3923897473364993},{"dataset":"urn_mavedb_00001217-a-3","value":0.4435481103956106},{"dataset":"urn_mavedb_00001227-a-1","value":0.2614118041416431},{"dataset":"urn_mavedb_00001239-a-1","value":0.3253252277651084},{"dataset":"urn_mavedb_00001243-a-1","value":-0.2690405663750969}]},{"id":"ESM3-1.4B","source_key":"ESM3-1.4B","label":"ESM3-1.4B","short_label":"ESM3-1.4B","mean":0.2553539991367496,"sem":0.05373793038766104,"source_mean":0.2553539991367496,"source_sem":0.05373793038766104,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.2553539991367496,"summary_max_abs_error":0.0,"name":"ESM3-1.4B","values":[{"dataset":"AAV","value":0.2731656511878863},{"dataset":"GB1","value":0.1830549657783},{"dataset":"GFP","value":0.6496042193663453},{"dataset":"TrpB","value":0.1710528364055714},{"dataset":"urn_mavedb_00000072-a-1","value":0.4264267971013317},{"dataset":"urn_mavedb_00000077-a-1","value":-0.0855964258690198},{"dataset":"urn_mavedb_00000077-b-1","value":0.0412365646388516},{"dataset":"urn_mavedb_00000079-a-1","value":0.2290950378190988},{"dataset":"urn_mavedb_00000090-a-1","value":0.533511521173392},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0641737628038998},{"dataset":"urn_mavedb_00000105-a-1","value":0.4057224232993942},{"dataset":"urn_mavedb_00000659-a-1","value":0.2802451648273279},{"dataset":"urn_mavedb_00000665-a-1","value":0.5315640874483013},{"dataset":"urn_mavedb_00001207-a-1","value":0.0716164755556888},{"dataset":"urn_mavedb_00001212-a-1","value":0.2186420056484085},{"dataset":"urn_mavedb_00001217-a-3","value":0.5223761272139376},{"dataset":"urn_mavedb_00001227-a-1","value":0.3465513504768664},{"dataset":"urn_mavedb_00001239-a-1","value":0.3385571424115663},{"dataset":"urn_mavedb_00001243-a-1","value":-0.220926198081106}]},{"id":"ESM-IF1","source_key":"ESM-IF1","label":"ESM-IF1","short_label":"ESM-IF1","mean":0.2579818735518448,"sem":0.04552624384056674,"source_mean":0.2579818735518448,"source_sem":0.04552624384056675,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.2579818735518448,"summary_max_abs_error":6.938893903907228e-18,"name":"ESM-IF1","values":[{"dataset":"AAV","value":0.377972476178167},{"dataset":"GB1","value":0.2571705523508572},{"dataset":"GFP","value":0.7051059856512725},{"dataset":"TrpB","value":0.1386509631059527},{"dataset":"urn_mavedb_00000072-a-1","value":0.3339446245724235},{"dataset":"urn_mavedb_00000077-a-1","value":0.0030548976557728},{"dataset":"urn_mavedb_00000077-b-1","value":0.114410404291308},{"dataset":"urn_mavedb_00000079-a-1","value":0.1582965497844785},{"dataset":"urn_mavedb_00000090-a-1","value":0.4310504147346933},{"dataset":"urn_mavedb_00000091-a-1","value":-0.0127113414784647},{"dataset":"urn_mavedb_00000105-a-1","value":0.4648148826925882},{"dataset":"urn_mavedb_00000659-a-1","value":0.3314962485756957},{"dataset":"urn_mavedb_00000665-a-1","value":0.407055577718779},{"dataset":"urn_mavedb_00001207-a-1","value":0.1010837456476274},{"dataset":"urn_mavedb_00001212-a-1","value":0.2618618201818122},{"dataset":"urn_mavedb_00001217-a-3","value":0.5045167586945125},{"dataset":"urn_mavedb_00001227-a-1","value":0.1918568196980193},{"dataset":"urn_mavedb_00001239-a-1","value":0.2106714144384137},{"dataset":"urn_mavedb_00001243-a-1","value":-0.0786471970088574}]},{"id":"ProTEUS","source_key":"ZEUS","label":"ProTEUS","short_label":"ProTEUS","mean":0.2708966009335355,"sem":0.05342189930450411,"source_mean":0.2708966009335355,"source_sem":0.05342189930450411,"n_assays":19,"n_observed":19,"n_missing":0,"mean_observed_only":0.2708966009335355,"summary_max_abs_error":0.0,"name":"ProTEUS","values":[{"dataset":"AAV","value":0.4408921299658648},{"dataset":"GB1","value":0.1016256755130329},{"dataset":"GFP","value":0.6510703826273097},{"dataset":"TrpB","value":0.1858601264938329},{"dataset":"urn_mavedb_00000072-a-1","value":0.3725220836036955},{"dataset":"urn_mavedb_00000077-a-1","value":0.1158764675080977},{"dataset":"urn_mavedb_00000077-b-1","value":0.2779636572729966},{"dataset":"urn_mavedb_00000079-a-1","value":0.2308749551858192},{"dataset":"urn_mavedb_00000090-a-1","value":0.5324417037406518},{"dataset":"urn_mavedb_00000091-a-1","value":-0.1189374305812662},{"dataset":"urn_mavedb_00000105-a-1","value":0.2068818132322859},{"dataset":"urn_mavedb_00000659-a-1","value":0.2300730972014606},{"dataset":"urn_mavedb_00000665-a-1","value":0.4877948603478242},{"dataset":"urn_mavedb_00001207-a-1","value":0.1055833068429008},{"dataset":"urn_mavedb_00001212-a-1","value":0.3996266166901676},{"dataset":"urn_mavedb_00001217-a-3","value":0.4733707444288703},{"dataset":"urn_mavedb_00001227-a-1","value":0.3734443856942312},{"dataset":"urn_mavedb_00001239-a-1","value":0.3974688159703464},{"dataset":"urn_mavedb_00001243-a-1","value":-0.3173979740009472}]}]};
ZERO_ASSAYS.models.sort((a,b)=>b.mean-a.mean);
const SELECTION_METRICS=[{"metric":"Spearman","n":13,"rows":[{"dataset":"urn_mavedb_00000072-a-1","ZEUS":"0.3725220836036955","MULTI-evolve":"0.3845950544806338"},{"dataset":"urn_mavedb_00000077-a-1","ZEUS":"0.1158764675080977","MULTI-evolve":"-0.0505044452048593"},{"dataset":"urn_mavedb_00000077-b-1","ZEUS":"0.2779636572729966","MULTI-evolve":"0.0402992535121914"},{"dataset":"urn_mavedb_00000079-a-1","ZEUS":"0.2308749551858192","MULTI-evolve":"0.2043366132418317"},{"dataset":"urn_mavedb_00000090-a-1","ZEUS":"0.5324417037406518","MULTI-evolve":"0.4691119967044051"},{"dataset":"urn_mavedb_00000091-a-1","ZEUS":"-0.1189374305812662","MULTI-evolve":"-0.0915683106876024"},{"dataset":"urn_mavedb_00000659-a-1","ZEUS":"0.2300730972014606","MULTI-evolve":"0.2856533504290556"},{"dataset":"urn_mavedb_00001207-a-1","ZEUS":"0.1055833068429008","MULTI-evolve":"0.1100312988199914"},{"dataset":"urn_mavedb_00001212-a-1","ZEUS":"0.3996266166901676","MULTI-evolve":"0.2511808369588203"},{"dataset":"urn_mavedb_00001217-a-3","ZEUS":"0.4733707444288703","MULTI-evolve":"0.4431934576011331"},{"dataset":"urn_mavedb_00001227-a-1","ZEUS":"0.3734443856942312","MULTI-evolve":"0.2823638665315434"},{"dataset":"urn_mavedb_00001239-a-1","ZEUS":"0.3974688159703464","MULTI-evolve":"0.2873313738022156"},{"dataset":"urn_mavedb_00001243-a-1","ZEUS":"-0.3173979740009472","MULTI-evolve":"-0.2418525891426623"}],"models":[{"name":"ProTEUS","mean":0.2363777253505403,"sem":0.06699053716592505,"values":[0.3725220836036955,0.1158764675080977,0.2779636572729966,0.2308749551858192,0.5324417037406518,-0.1189374305812662,0.2300730972014606,0.1055833068429008,0.3996266166901676,0.4733707444288703,0.3734443856942312,0.3974688159703464,-0.3173979740009472]},{"name":"MULTI-evolve","mean":0.1826285966958998,"sem":0.060058679871137714,"values":[0.3845950544806338,-0.0505044452048593,0.0402992535121914,0.2043366132418317,0.4691119967044051,-0.0915683106876024,0.2856533504290556,0.1100312988199914,0.2511808369588203,0.4431934576011331,0.2823638665315434,0.2873313738022156,-0.2418525891426623]}],"wilcoxon_one_sided_p":0.047119140625},{"metric":"AUROC","n":13,"rows":[{"dataset":"urn_mavedb_00000072-a-1","ZEUS":"0.7461160518225736","MULTI-evolve":"0.7678757685551164"},{"dataset":"urn_mavedb_00000077-a-1","ZEUS":"0.5875594860171174","MULTI-evolve":"0.4702512023755095"},{"dataset":"urn_mavedb_00000077-b-1","ZEUS":"0.6214618114401821","MULTI-evolve":"0.5064799816061533"},{"dataset":"urn_mavedb_00000079-a-1","ZEUS":"0.6272330097170318","MULTI-evolve":"0.6115630508469364"},{"dataset":"urn_mavedb_00000090-a-1","ZEUS":"0.8497193204513886","MULTI-evolve":"0.7828647806754391"},{"dataset":"urn_mavedb_00000091-a-1","ZEUS":"0.3672161172161172","MULTI-evolve":"0.3489010989010989"},{"dataset":"urn_mavedb_00000659-a-1","ZEUS":"0.6262992793656162","MULTI-evolve":"0.6605052779187103"},{"dataset":"urn_mavedb_00001207-a-1","ZEUS":"0.5436112259834576","MULTI-evolve":"0.5479946269554267"},{"dataset":"urn_mavedb_00001212-a-1","ZEUS":"0.8081298936872441","MULTI-evolve":"0.732725091927773"},{"dataset":"urn_mavedb_00001217-a-3","ZEUS":"0.7634280140654316","MULTI-evolve":"0.7372354965768319"},{"dataset":"urn_mavedb_00001227-a-1","ZEUS":"0.6769162640901771","MULTI-evolve":"0.6366481912813434"},{"dataset":"urn_mavedb_00001239-a-1","ZEUS":"0.7232756494949477","MULTI-evolve":"0.6558021611273199"},{"dataset":"urn_mavedb_00001243-a-1","ZEUS":"0.3436283169031473","MULTI-evolve":"0.3763732610960373"}],"models":[{"name":"ProTEUS","mean":0.6372764954041871,"sem":0.04251157892904456,"values":[0.7461160518225736,0.5875594860171174,0.6214618114401821,0.6272330097170318,0.8497193204513886,0.3672161172161172,0.6262992793656162,0.5436112259834576,0.8081298936872441,0.7634280140654316,0.6769162640901771,0.7232756494949477,0.3436283169031473]},{"name":"MULTI-evolve","mean":0.6027092299879767,"sem":0.039837197354943114,"values":[0.7678757685551164,0.4702512023755095,0.5064799816061533,0.6115630508469364,0.7828647806754391,0.3489010989010989,0.6605052779187103,0.5479946269554267,0.732725091927773,0.7372354965768319,0.6366481912813434,0.6558021611273199,0.3763732610960373]}],"wilcoxon_one_sided_p":0.0286865234375},{"metric":"AUPR","n":13,"rows":[{"dataset":"urn_mavedb_00000072-a-1","ZEUS":"0.4428843669530343","MULTI-evolve":"0.5125029151297098"},{"dataset":"urn_mavedb_00000077-a-1","ZEUS":"0.7696940087089706","MULTI-evolve":"0.6878863288297244"},{"dataset":"urn_mavedb_00000077-b-1","ZEUS":"0.6471486088091845","MULTI-evolve":"0.5416556957484603"},{"dataset":"urn_mavedb_00000079-a-1","ZEUS":"0.615106389496255","MULTI-evolve":"0.5995562761906468"},{"dataset":"urn_mavedb_00000090-a-1","ZEUS":"0.937396458785118","MULTI-evolve":"0.8987900410937021"},{"dataset":"urn_mavedb_00000091-a-1","ZEUS":"0.2286611259556546","MULTI-evolve":"0.2298017356372231"},{"dataset":"urn_mavedb_00000659-a-1","ZEUS":"0.7186777931065003","MULTI-evolve":"0.7225712905157805"},{"dataset":"urn_mavedb_00001207-a-1","ZEUS":"0.5132607153867859","MULTI-evolve":"0.5175403533744222"},{"dataset":"urn_mavedb_00001212-a-1","ZEUS":"0.8916240079119209","MULTI-evolve":"0.8486910022395282"},{"dataset":"urn_mavedb_00001217-a-3","ZEUS":"0.8376117419612354","MULTI-evolve":"0.8183193167700125"},{"dataset":"urn_mavedb_00001227-a-1","ZEUS":"0.6170320782187433","MULTI-evolve":"0.5656022619783797"},{"dataset":"urn_mavedb_00001239-a-1","ZEUS":"0.7744841379638254","MULTI-evolve":"0.7045122838549565"},{"dataset":"urn_mavedb_00001243-a-1","ZEUS":"0.4385255883593885","MULTI-evolve":"0.4571224287760417"}],"models":[{"name":"ProTEUS","mean":0.6486236170474322,"sem":0.056435065293215445,"values":[0.4428843669530343,0.7696940087089706,0.6471486088091845,0.615106389496255,0.937396458785118,0.2286611259556546,0.7186777931065003,0.5132607153867859,0.8916240079119209,0.8376117419612354,0.6170320782187433,0.7744841379638254,0.4385255883593885]},{"name":"MULTI-evolve","mean":0.6234270715491222,"sem":0.05075237524997169,"values":[0.5125029151297098,0.6878863288297244,0.5416556957484603,0.5995562761906468,0.8987900410937021,0.2298017356372231,0.7225712905157805,0.5175403533744222,0.8486910022395282,0.8183193167700125,0.5656022619783797,0.7045122838549565,0.4571224287760417]}],"wilcoxon_one_sided_p":0.047119140625}];
const TAU=Math.PI*2;
const palette={ProTEUS:C.blue,EVOLVEpro:C.rose,Additive:C.teal,'MULTI-evolve':'#778A9B',AlphaFold3:'#778A9B',ProteinNPT:C.rose};
const lerp=(a,b,p)=>a+(b-a)*p;
function t(c,s,x,y,size=28,col=C.ink,align='left',weight=400){text(c,s,x,y,size,col,weight,align);}
function path(c,pts,col,width=3,dash=[]){if(!pts.length)return;c.beginPath();pts.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.lineWidth=width;c.strokeStyle=col;c.lineCap='round';c.lineJoin='round';c.setLineDash(dash);c.stroke();c.setLineDash([]);}
function dot(c,x,y,r,col){c.beginPath();c.arc(x,y,r,0,TAU);c.fillStyle=col;c.fill();}
function reveal(c,p,fn){c.save();c.globalAlpha*=clamp(p);fn();c.restore();}
function lineProgress(c,pts,p,col,w=4,dash=[]){if(!pts.length||p<=0)return;const z=clamp(p)*(pts.length-1),i=Math.floor(z),q=pts.slice(0,i+1);if(i<pts.length-1)q.push([lerp(pts[i][0],pts[i+1][0],z-i),lerp(pts[i][1],pts[i+1][1],z-i)]);path(c,q,col,w,dash);}
function points(c,pts,p,col,r=6){pts.forEach(([x,y],i)=>reveal(c,clamp((p-i/Math.max(1,pts.length-1))*.8+1),()=>{if(p*pts.length>=i+.3)dot(c,x,y,r,col);}));}
function legend(c,items,x,y,spacing=260,size=28){items.forEach(([name,col],i)=>{const xx=x+i*spacing;path(c,[[xx,y],[xx+31,y]],col,5);dot(c,xx+16,y,5,col);t(c,name,xx+43,y,size,col);});}
function axes(c,b,xticks,yticks,xlabel,ylabel,opts={}){const [x,y,w,h]=b;
  yticks.forEach(([v,lab])=>{const yy=y+h*(1-v);path(c,[[x,yy],[x+w,yy]],C.grid,1);t(c,lab,x-17,yy,27,C.muted,'right');});
  path(c,[[x,y],[x,y+h],[x+w,y+h]],C.axis,2);
  xticks.forEach(([v,lab])=>{const xx=x+w*v;path(c,[[xx,y+h],[xx,y+h+7]],C.axis,2);t(c,lab,xx,y+h+30,27,C.muted,'center');});
  if(xlabel)t(c,xlabel,x+w/2,y+h+(opts.xgap||65),28,C.muted,'center');
  if(ylabel)t(c,ylabel,x,y-31,29,C.ink);
}
function band(c,b,rs,xmap,ymap,p,col){if(!rs.length||p<=0)return;c.save();c.beginPath();c.rect(b[0]-1,b[1]-1,(b[2]+2)*clamp(p),b[3]+2);c.clip();c.globalAlpha*=.13;c.beginPath();rs.forEach((r,i)=>i?c.lineTo(xmap(r.x),ymap(r.y+r.sem)):c.moveTo(xmap(r.x),ymap(r.y+r.sem)));[...rs].reverse().forEach(r=>c.lineTo(xmap(r.x),ymap(r.y-r.sem)));c.closePath();c.fillStyle=col;c.fill();c.restore();}
function seriesChart(c,box,p,rows,config){const[x,y,w,h]=box;const names=[...new Set(rows.map(r=>r.name))],top=names.length>2?135:98;const b=[x+98,y+top,w-140,h-top-109];const xmin=config.xmin,xmax=config.xmax,ymin=config.ymin,ymax=config.ymax;
  const xf=v=>config.logx?(Math.log(v)-Math.log(xmin))/(Math.log(xmax)-Math.log(xmin)):(v-xmin)/(xmax-xmin);
  const xm=v=>b[0]+b[2]*xf(v),ym=v=>b[1]+b[3]*(1-(v-ymin)/(ymax-ymin));
  axes(c,b,config.xticks.map(v=>[xf(v),String(v)]),config.yticks.map(v=>[(v-ymin)/(ymax-ymin),config.percent?Math.round(v*100)+'%':v.toFixed(config.decimals??2)]),config.xlabel,config.ylabel);
  if(config.breakY){path(c,[[b[0]-6,b[1]+b[3]-5],[b[0]+6,b[1]+b[3]-12]],C.ink,2);path(c,[[b[0]-6,b[1]+b[3]-12],[b[0]+6,b[1]+b[3]-19]],C.ink,2);}
  if(names.length<=2)legend(c,names.map(n=>[n,palette[n]]),x+100,y+18,w>900?330:270,28);
  else {legend(c,names.slice(0,2).map(n=>[n,palette[n]]),x+105,y+4,350,27);legend(c,names.slice(2).map(n=>[n,palette[n]]),x+105,y+42,350,27);}
  names.slice().reverse().forEach(name=>{const rs=rows.filter(r=>r.name===name).sort((a,b)=>a.x-b.x),col=palette[name],pts=rs.map(r=>[xm(r.x),ym(r.y)]);band(c,b,rs,xm,ym,p,col);lineProgress(c,pts,p,col,name==='ProTEUS'?5:3.5);points(c,pts,p,col,name==='ProTEUS'?7:5.5);});
  t(c,config.note||'Mean ± SEM across datasets',x+w-40,y+h-9,24,C.muted,'right');
}
function scaling(c,box,p){const[x,y,w,h]=box,s=DATA.scaling;const bx=x+105,bw=w-150,ph=(h-205)/2;const top=y+47,bot=top+ph+76;
  const xx=v=>bx+bw*(v-s.source_bounds.x[0])/(s.source_bounds.x[1]-s.source_bounds.x[0]);
  const xp=i=>(s.source_x[i]-s.source_bounds.x[0])/(s.source_bounds.x[1]-s.source_bounds.x[0]);
  const ticks=[0,2,4,7].map(i=>[xp(i),String(s.fractions[i])]);
  const yt=(v,lo)=>1-(v-lo)/65;
  axes(c,[bx,top,bw,ph],[],s.mse_ticks.map(([v,yy])=>[yt(yy,56),String(v)]),'','Test MSE · logarithmic scale');
  axes(c,[bx,bot,bw,ph],ticks,s.rho_ticks.map(([v,yy])=>[yt(yy,145),v.toFixed(2)]),'Training-data fraction (%) · logarithmic scale','Test Spearman ρ',{xgap:70});
  const a=s.source_x.map((v,i)=>[xx(v),top+ph*(s.source_mse_y[i]-56)/65]);
  const b=s.source_x.map((v,i)=>[xx(v),bot+ph*(s.source_rho_y[i]-145)/65]);
  const fit=s.fit_source_endpoints.map(([sx,sy])=>[xx(sx),top+ph*(sy-56)/65]);
  lineProgress(c,fit,p,C.gold,2.5,[9,8]);lineProgress(c,a,p,C.blue,4.5);points(c,a,p,C.blue,7);lineProgress(c,b,p,C.blue,4.5);points(c,b,p,C.blue,7);
  reveal(c,p,()=>{t(c,'Power-law exponent: −0.187',bx+bw-10,top+27,26,C.gold,'right');t(c,'R² = 0.989',bx+bw-10,top+62,25,C.gold,'right');});
}
function zeroSummary(c,box,p,compact=false){
  const[x,y,w,h]=box,fs=compact?12:22,rows=ZERO_ASSAYS.models;
  const left=x+(compact?164:205),right=x+w-24,top=y+55,bottom=y+h-60;
  const xm=v=>left+(right-left)*v/.35;
  t(c,'19 held-out assays · 15 models',x+6,y+12,compact?15:26,C.ink,'left',600);
  [0,.1,.2,.3].forEach(v=>{path(c,[[xm(v),top-10],[xm(v),bottom+12]],C.grid,1);t(c,v.toFixed(1),xm(v),bottom+32,fs,C.muted,'center');});
  rows.forEach((r,i)=>{const yy=top+(bottom-top)*i/14,col=i?C.muted:C.blue,q=clamp(p*1.8-i*.035);
    t(c,r.name,x+6,yy,fs,col,'left',i?400:600);
    reveal(c,q,()=>{path(c,[[xm(r.mean-r.sem),yy],[xm(r.mean+r.sem),yy]],col,compact?1.5:2.5);[r.mean-r.sem,r.mean+r.sem].forEach(v=>path(c,[[xm(v),yy-5],[xm(v),yy+5]],col,1.3));dot(c,xm(r.mean),yy,i?(compact?3:5):(compact?4:7),col);});
  });
  t(c,'Mean Spearman ρ ± SEM',compact?x+w/2:left+(right-left)/2,y+h-9,fs,C.muted,'center');
}
function heatColor(value){
  if(value===null)return '#eceef0';
  const lo=value<0?[178,91,98]:[248,250,251],hi=value<0?[248,250,251]:[49,114,157];
  const q=value<0?clamp((value+.4)/.4):clamp(value/.8);
  return 'rgb('+lo.map((v,i)=>Math.round(v+(hi[i]-v)*q)).join(',')+')';
}
function zeroHeatmap(c,box,p,all=false,compact=false){
  const[x,y,w,h]=box,models=ZERO_ASSAYS.models.slice(0,all?15:5),fs=compact?12:19;
  const left=x+(compact?96:174),right=x+w-(all?74:14),top=y+(all?150:100),bottom=y+h-60;
  const cw=(right-left)/models.length,rh=(bottom-top)/19;
  t(c,all?'19 assays × 15 models':'19 assays · five leading models',x+6,y+12,compact?16:27,C.ink,'left',600);
  models.forEach((model,i)=>{
    const col=model.name==='ProTEUS'?C.blue:C.muted;
    if(all){c.save();c.translate(left+cw*(i+.5),top-12);c.rotate(-Math.PI/3);t(c,model.name,0,0,fs,col,'left');c.restore();}
    else {
      const labels=model.name==='MSA Transformer'?(compact?['MSA','Transf.']:['MSA','Transformer']):model.name.startsWith('ESM3')?['ESM3','1.4B']:model.name.startsWith('SaProt')?['SaProt','650M']:[model.name];
      labels.forEach((label,j)=>t(c,label,left+cw*(i+.5),top-38+j*20,compact?10:17,col,'center',model.name==='ProTEUS'?600:400));
    }
  });
  ZERO_ASSAYS.assays.forEach((a,j)=>{
    t(c,a.short_label||a.label,left-12,top+(j+.5)*rh,fs,C.muted,'right');
    models.forEach((m,i)=>{const value=m.values.find(v=>v.dataset===a.id).value,q=clamp(p*1.7-j*.025);
      reveal(c,q,()=>{c.fillStyle=heatColor(value);c.fillRect(left+i*cw+1,top+j*rh+1,cw-2,rh-2);
        if(value===null){path(c,[[left+i*cw+cw*.35,top+(j+.3)*rh],[left+i*cw+cw*.65,top+(j+.7)*rh]],C.muted,1);}
      });
    });
  });
  const gw=Math.min(210,w*.38),gx=left,gy=bottom+25;
  for(let i=0;i<gw;i++){c.fillStyle=heatColor(-.4+1.2*i/gw);c.fillRect(gx+i,gy,1,9);}
  [-.4,0,.4,.8].forEach(v=>t(c,v.toFixed(1),gx+(v+.4)/1.2*gw,gy+24,compact?10:17,C.muted,'center'));
  t(c,'Spearman ρ',right,gy+5,compact?11:19,C.muted,'right');
}
function selectionQuality(c,box,p,mini=false){
  const[x,y,w,h]=box,fs=mini?17:25,top=y+12,bottom=y+h-42,plotH=bottom-top;
  // Figure 2F uses distinct, nonzero axis ranges. The double slash marks
  // the omitted baseline; means and SEM are the original source values.
  const axes=[{lo:.05,hi:.35,ticks:[.1,.2,.3]},{lo:.52,hi:.72,ticks:[.6,.7]},{lo:.5,hi:.76,ticks:[.6,.7]}];
  SELECTION_METRICS.forEach((metric,i)=>{
    const cell=w/3,left=x+cell*i+54,right=x+cell*(i+1)-16,center=(left+right)/2,{lo,hi,ticks}=axes[i];
    const ym=v=>bottom-plotH*(v-lo)/(hi-lo);
    ticks.forEach(v=>{path(c,[[left,ym(v)],[right,ym(v)]],C.grid,1);t(c,v.toFixed(1),left-9,ym(v),fs-2,C.muted,'right');});
    path(c,[[left,top],[left,bottom],[right,bottom]],C.axis,1.3);
    path(c,[[left-5,bottom-5],[left+5,bottom-11]],C.ink,1.7);
    path(c,[[left-5,bottom-11],[left+5,bottom-17]],C.ink,1.7);
    const bw=Math.min(32,(right-left)*.24);
    metric.models.forEach((m,j)=>{
      const cx=center+(j?1:-1)*bw*.7,col=j?'#85919B':'#4383AB',q=clamp(p*1.5-i*.12);
      reveal(c,q,()=>{c.fillStyle=j?'#DADEE1':'#8BBBD5';c.fillRect(cx-bw/2,ym(m.mean),bw,bottom-ym(m.mean));path(c,[[cx-bw/2,bottom],[cx-bw/2,ym(m.mean)],[cx+bw/2,ym(m.mean)],[cx+bw/2,bottom]],col,1.8);path(c,[[cx,ym(m.mean-m.sem)],[cx,ym(m.mean+m.sem)]],col,2);[m.mean-m.sem,m.mean+m.sem].forEach(v=>path(c,[[cx-5,ym(v)],[cx+5,ym(v)]],col,1.5));});
    });
    t(c,metric.metric==='Spearman'?'Spearman ρ':metric.metric,center,bottom+27,fs,C.ink,'center');
  });
}
function nomination(c,box,p,mini=false){
  const[x,y,w,h]=box,fs=mini?18:25,b=[x+86,y+38,w-111,h-107];
  const lo=.55,hi=.85,xf=v=>Math.log(v/5)/Math.log(20),xm=v=>b[0]+b[2]*xf(v),ym=v=>b[1]+b[3]*(1-(v-lo)/(hi-lo));
  t(c,'Fraction at or above assay mean',b[0],y+5,fs,C.ink);
  [.6,.7,.8].forEach(v=>{path(c,[[b[0],ym(v)],[b[0]+b[2],ym(v)]],C.grid,1);t(c,Math.round(v*100)+'%',b[0]-12,ym(v),fs,C.muted,'right');});
  path(c,[[b[0],b[1]],[b[0],b[1]+b[3]],[b[0]+b[2],b[1]+b[3]]],C.axis,1.5);
  path(c,[[b[0]-5,b[1]+b[3]-5],[b[0]+5,b[1]+b[3]-11]],C.ink,1.7);
  path(c,[[b[0]-5,b[1]+b[3]-11],[b[0]+5,b[1]+b[3]-17]],C.ink,1.7);
  [5,10,15,24,50,100].forEach(v=>t(c,String(v),xm(v),b[1]+b[3]+24,fs,C.muted,'center'));
  ['MULTI-evolve','ProTEUS'].forEach(name=>{const rs=DATA.selection.filter(r=>r.name===name),col=name==='ProTEUS'?'#4383AB':'#85919B',pts=rs.map(r=>[xm(r.x),ym(r.y)]);band(c,b,rs,xm,ym,p,col);lineProgress(c,pts,p,col,name==='ProTEUS'?3.7:2.6,name==='ProTEUS'?[]:[7,4]);points(c,pts,p,col,5);});
  t(c,c.proteusCssUnit&&w<=600?'Selection budget K · log scale':'Selection budget K · logarithmic scale',b[0]+b[2]/2,y+h-16,fs,C.muted,'center');
}
function selectionPanel(c,box,p){
  const[x,y,w,h]=box;
  t(c,'First-round variant selection',x+8,y+12,26,C.ink,'left',600);
  t(c,'13 held-out single-mutant assays',x+8,y+45,21,C.muted);
  legend(c,[['ProTEUS','#4383AB'],['MULTI-evolve','#85919B']],x+20,y+83,w*.47,21);
  selectionQuality(c,[x+4,y+103,w-8,h*.33],p,true);
  nomination(c,[x,y+h*.52,w,h*.48],p,true);
}
function zero(c,box,p){
  const[x,y,w,h]=box;
  zeroSummary(c,[x,y,w*.45,h],p);
  path(c,[[x+w*.47,y+10],[x+w*.47,y+h-20]],C.grid,1);
  selectionPanel(c,[x+w*.49,y,w*.51,h],p);
}

function fullZero(c,box,p){zeroHeatmap(c,box,p,true);}

function fine(c,box,p){const[x,y,w,h]=box,left=x+225,right=x+w-28,top=y+82,bottom=y+h-110;const xm=v=>left+(right-left)*(v+.2)/1.2;
  t(c,'Supervised fivefold CV · 19 assays',x+6,y+6,27,C.muted);
  [-.2,0,.5,1].forEach(v=>{path(c,[[xm(v),top-25],[xm(v),bottom+25]],C.grid,1);t(c,v.toFixed(1),xm(v),bottom+58,26,C.muted,'center');});
  DATA.fine.forEach((r,i)=>{const yy=top+52+i*(bottom-top-94),col=palette[r.name],q=clamp(p*1.45-i*.17);t(c,r.name,x+5,yy,31,col);
    reveal(c,q,()=>{r.values.forEach((v,j)=>{if(v.value!==null){const yyj=yy+((j*7)%19-9)*4.2;c.save();c.globalAlpha*=.40;dot(c,xm(v.value),yyj,4,col);c.restore();}});path(c,[[xm(r.mean-r.sem),yy],[xm(r.mean+r.sem),yy]],col,5);[r.mean-r.sem,r.mean+r.sem].forEach(v=>path(c,[[xm(v),yy-14],[xm(v),yy+14]],col,3));dot(c,xm(r.mean),yy,11,col);t(c,r.mean.toFixed(3),xm(r.mean),yy-63,30,col,'center',600);});});
  t(c,'Mean Spearman ρ ± SEM',left+(right-left)/2,y+h-25,28,C.muted,'center');
  t(c,'Small dots: individual assays',x+7,y+40,23,C.muted);
}
function quality(c,box,p){const[x,y,w,h]=box,b=[x+91,y+75,w-130,h-180],ym=v=>b[1]+b[3]*(1-v/1.05),names=['Complex LDDT','Prot-Prot iLDDT','TM-score','Physical Validity'],labels=[['Complex','LDDT'],['Interface','LDDT'],['TM-score',''],['Validity','7 checks']];
  axes(c,b,[],[0,.5,1].map(v=>[v/1.05,v.toFixed(1)]),'','CASP15 · mean scores');
  names.forEach((metric,i)=>{const center=b[0]+b[2]*(i+.5)/4,bw=54;['AlphaFold3','ProTEUS'].forEach((name,j)=>{const r=DATA.quality.find(d=>d.name===name&&d.metric===metric),q=clamp(p*1.55-i*.10-j*.03),xx=center+(j?7:-bw-7),col=name==='ProTEUS'?C.structure:palette[name],v=r.mean*q;c.fillStyle=col;c.fillRect(xx,ym(v),bw,ym(0)-ym(v));reveal(c,q,()=>t(c,r.mean.toFixed(2),xx+bw/2,ym(r.mean)-22,25,col,'center'));});t(c,labels[i][0],center,b[1]+b[3]+31,27,C.muted,'center');t(c,labels[i][1],center,b[1]+b[3]+65,25,C.muted,'center');});
  t(c,'CASP15 benchmark · means shown',x+w-30,y+h-3,24,C.muted,'right');
}
function qualityLegend(c,box){const[x,y,w,h]=box;legend(c,[['AlphaFold 3 · 200 steps','#778A9B'],['Our predictor · 2 steps',C.structure]],x+8,y+h/2,Math.max(430,w*.50),28);}
function runtime(c,box,p){const[x,y,w,h]=box,b=[x+105,y+95,w-145,h-205],ymin=.45,ymax=1000;const ym=v=>b[1]+b[3]*(1-(Math.log10(v)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin))),xm=i=>b[0]+b[2]*(i/4),bins=['<1k','1–2k','2–3k','3–4k','≥4k'];
  axes(c,b,bins.map((s,i)=>[i/4,s]),[1,10,100,1000].map(v=>[(Math.log10(v)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)),String(v)]),'Atom-count bin','Mean inference time (s) · log scale');
  legend(c,[['Our predictor',C.structure],['AlphaFold 3',palette.AlphaFold3]],x+107,y+15,350,28);
  ['AlphaFold3','ProTEUS'].forEach(name=>{const rs=DATA.runtime.filter(r=>r.name===name),pts=rs.map((r,i)=>[xm(i),ym(r.seconds)]);lineProgress(c,pts,p,name==='ProTEUS'?C.structure:palette[name],name==='ProTEUS'?5:4);points(c,pts,p,name==='ProTEUS'?C.structure:palette[name],7);});
  const our=DATA.runtime.filter(r=>r.name==='ProTEUS'),af=DATA.runtime.filter(r=>r.name==='AlphaFold3');our.forEach((r,i)=>reveal(c,clamp((p-.12*i)*3),()=>t(c,(af[i].seconds/r.seconds).toFixed(1)+'×',xm(i),ym(Math.sqrt(r.seconds*af[i].seconds)),28,C.structure,'center',600)));
  t(c,'NVIDIA RTX A6000 · denoising + validity enforcement',x+w-25,y+h-8,24,C.muted,'right');
}
function higher(c,box,p){const[x,y,w,h]=box,b=[x+103,y+101,w-146,h-216],lo=-.25,hi=1,ym=v=>b[1]+b[3]*(1-(v-lo)/(hi-lo));
  axes(c,b,[],[-.25,0,.5,1].map(v=>[(v-lo)/(hi-lo),v.toFixed(v===-.25?2:1)]),'','Test Spearman ρ');
  legend(c,[['ProTEUS',C.blue],['MULTI-evolve',palette['MULTI-evolve']]],x+105,y+19,330,28);
  [1,2,3].forEach((k,i)=>{const rs=DATA.higher.filter(r=>r.k===k),center=b[0]+b[2]*(i+.5)/3;[['ours',C.blue],['baseline',palette['MULTI-evolve']]].forEach(([key,col],j)=>{const vals=rs.map(r=>r[key]),mean=vals.reduce((a,v)=>a+v,0)/vals.length,sem=Math.sqrt(vals.reduce((s,v)=>s+(v-mean)**2,0)/(vals.length-1)/vals.length),xx=center+(j?8:-85),bw=77,q=clamp(p*1.5-i*.13-j*.035);c.save();c.globalAlpha*=.32;c.fillStyle=col;c.fillRect(xx,ym(mean*q),bw,ym(0)-ym(mean*q));c.restore();reveal(c,q,()=>{path(c,[[xx+bw/2,ym(mean-sem)],[xx+bw/2,ym(mean+sem)]],col,3);[mean-sem,mean+sem].forEach(v=>path(c,[[xx+bw/2-12,ym(v)],[xx+bw/2+12,ym(v)]],col,3));vals.forEach((v,jj)=>dot(c,xx+12+((jj*11)%rs.length)/(rs.length-1)*(bw-24),ym(v),4.8,col));path(c,[[xx,ym(mean)],[xx+bw,ym(mean)]],col,5);});});t(c,'K = '+k,center,b[1]+b[3]+33,29,C.ink,'center');t(c,rs.length+' comparisons',center,b[1]+b[3]+68,25,C.muted,'center');});
  t(c,'Train on ≤K substitutions → test on >K',x+w/2,y+h-7,28,C.muted,'center');
}
// Narrow screens use CSS-pixel coordinates and a dedicated layout, so labels
// remain readable instead of shrinking a 1280 px chart into a phone viewport.
function layoutForWidth(cssWidth,name){
  return cssWidth<600
    ? {width:Math.max(240,Math.round(cssWidth)),height:['Fig2E','ZeroAssays','SelectionQuality'].includes(name)?650:400}
    : {width:1280,height:760};
}

function compactLegend(c,items,x,y,width){
  items.forEach(([name,col],i)=>{
    const xx=x+(i%2)*width/2,yy=y+Math.floor(i/2)*21;
    path(c,[[xx,yy],[xx+14,yy]],col,2);
    dot(c,xx+7,yy,2.5,col);
    t(c,name,xx+20,yy,12,col);
  });
}

function compactAxes(c,b,xticks,yticks,xlabel,ylabel){
  const[x,y,w,h]=b;
  yticks.forEach(([v,label])=>{
    const yy=y+h*(1-v);
    path(c,[[x,yy],[x+w,yy]],C.grid,1);
    t(c,label,x-9,yy,12,C.muted,'right');
  });
  path(c,[[x,y],[x,y+h],[x+w,y+h]],C.axis,1);
  xticks.forEach(([v,label])=>t(c,label,x+w*v,y+h+19,12,C.muted,'center'));
  if(xlabel)t(c,xlabel,x+w/2,y+h+43,13,C.muted,'center');
  if(ylabel)t(c,ylabel,x,y-19,13,C.ink);
}

function compactScaling(c,box,p){
  const[x,y,w,h]=box,s=DATA.scaling,left=x+60,width=w-83;
  const plotHeight=(h-163)/2,upper=[left,y+42,width,plotHeight];
  const lower=[left,y+109+plotHeight,width,plotHeight];
  const xf=v=>(v-s.source_bounds.x[0])/(s.source_bounds.x[1]-s.source_bounds.x[0]);
  const xm=v=>left+width*xf(v),yn=(v,lo)=>1-(v-lo)/65;
  compactAxes(c,upper,[],s.mse_ticks.map(([v,sy])=>[yn(sy,56),String(v)]),'','Test MSE · log scale');
  compactAxes(c,lower,[0,2,4,7].map(i=>[xf(s.source_x[i]),String(s.fractions[i])]),s.rho_ticks.map(([v,sy])=>[yn(sy,145),v.toFixed(2)]),'Training-data fraction (%) · log scale','Test Spearman ρ');
  const mse=s.source_x.map((v,i)=>[xm(v),upper[1]+plotHeight*(s.source_mse_y[i]-56)/65]);
  const rho=s.source_x.map((v,i)=>[xm(v),lower[1]+plotHeight*(s.source_rho_y[i]-145)/65]);
  const fit=s.fit_source_endpoints.map(([sx,sy])=>[xm(sx),upper[1]+plotHeight*(sy-56)/65]);
  lineProgress(c,fit,p,C.gold,1.3,[5,4]);
  lineProgress(c,mse,p,C.blue,2.2);points(c,mse,p,C.blue,3.5);
  lineProgress(c,rho,p,C.blue,2.2);points(c,rho,p,C.blue,3.5);
  reveal(c,p,()=>{
    t(c,'Exponent −0.187',x+w-23,upper[1]+13,12,C.gold,'right');
    t(c,'R² = 0.989',x+w-23,upper[1]+32,12,C.gold,'right');
  });
}

function compactZero(c,box,p){zeroSummary(c,box,p,true);}
function compactAssays(c,box,p){
  // Full model names remain readable; horizontal overflow is contained in this chart.
  const[x,y,w,h]=box;
  zeroHeatmap(c,[x,y,w,h],p,false,true);
}
function compactQualityPanel(c,box,p){
  const[x,y,w,h]=box;
  c.save();c.translate(x,y);const scale=w/600;c.scale(scale,scale);
  selectionPanel(c,[0,0,600,h/scale],p);c.restore();
}
function compactSelection(c,box,p){
  const[x,y,w,h]=box,b=[x+60,y+82,w-84,h-160],xm=v=>b[0]+b[2]*Math.log(v/5)/Math.log(20),ym=v=>b[1]+b[3]*(1-(v-.55)/.30);
  compactLegend(c,[['ProTEUS',C.blue],['MULTI-evolve',palette['MULTI-evolve']]],x+15,y+20,w-30);
  compactAxes(c,b,[5,10,24,50,100].map(v=>[Math.log(v/5)/Math.log(20),String(v)]),[.6,.7,.8].map(v=>[(v-.55)/.30,Math.round(v*100)+'%']),'Top K selected variants','At or above assay mean');
  path(c,[[b[0]-5,b[1]+b[3]-4],[b[0]+5,b[1]+b[3]-10]],C.ink,1.5);path(c,[[b[0]-5,b[1]+b[3]-10],[b[0]+5,b[1]+b[3]-16]],C.ink,1.5);
  ['MULTI-evolve','ProTEUS'].forEach(name=>{const rs=DATA.selection.filter(r=>r.name===name),col=palette[name];band(c,b,rs,xm,ym,p,col);const pts=rs.map(r=>[xm(r.x),ym(r.y)]);lineProgress(c,pts,p,col,2);points(c,pts,p,col,3.5);});
  t(c,'13 assays · mean ± SEM',x+w/2,y+h-15,12,C.muted,'center');
}

function compactFine(c,box,p){
  const[x,y,w,h]=box,left=x+30,right=x+w-24,bottom=y+h-65;
  const xm=v=>left+(right-left)*(v+.2)/1.2;
  t(c,'Fivefold CV · 19 held-out assays',x+14,y+18,13,C.muted);
  t(c,'Small dots show individual assays',x+14,y+39,12,C.muted);
  [-.2,0,.5,1].forEach(v=>{
    path(c,[[xm(v),y+59],[xm(v),bottom]],C.grid,1);
    t(c,v.toFixed(1),xm(v),bottom+20,12,C.muted,'center');
  });
  DATA.fine.forEach((r,i)=>{
    const yy=y+124+i*(h-142)/2,col=palette[r.name];
    t(c,r.name,left,yy-51,15,col,'left',600);
    t(c,r.mean.toFixed(3),right,yy-51,16,col,'right',600);
    reveal(c,clamp(p*1.45-i*.17),()=>{
      r.values.forEach((v,j)=>{
        if(v.value===null)return;
        c.save();c.globalAlpha*=.45;
        dot(c,xm(v.value),yy+((j*7)%19-9)*2.6,2.5,col);
        c.restore();
      });
      path(c,[[xm(r.mean-r.sem),yy],[xm(r.mean+r.sem),yy]],col,2.5);
      [r.mean-r.sem,r.mean+r.sem].forEach(v=>path(c,[[xm(v),yy-7],[xm(v),yy+7]],col,1.5));
      dot(c,xm(r.mean),yy,6,col);
    });
  });
  t(c,'Mean Spearman ρ ± SEM',x+w/2,y+h-18,13,C.muted,'center');
}

function compactRuntime(c,box,p){
  const[x,y,w,h]=box,b=[x+60,y+81,w-83,h-156],lo=.45,hi=1000;
  const yf=v=>(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo));
  const ym=v=>b[1]+b[3]*(1-yf(v)),xm=i=>b[0]+b[2]*i/4;
  compactLegend(c,[['Our predictor',C.structure],['AlphaFold 3',palette.AlphaFold3]],x+15,y+22,w-30);
  compactAxes(c,b,['<1k','1–2k','2–3k','3–4k','≥4k'].map((v,i)=>[i/4,v]),[1,10,100,1000].map(v=>[yf(v),String(v)]),'Atom-count bin','Inference time (s) · log scale');
  ['AlphaFold3','ProTEUS'].forEach(name=>{
    const rs=DATA.runtime.filter(r=>r.name===name),pts=rs.map((r,i)=>[xm(i),ym(r.seconds)]);
    lineProgress(c,pts,p,name==='ProTEUS'?C.structure:palette[name],name==='ProTEUS'?2.5:1.7);
    points(c,pts,p,name==='ProTEUS'?C.structure:palette[name],3.6);
  });
  const ours=DATA.runtime.filter(r=>r.name==='ProTEUS'),af=DATA.runtime.filter(r=>r.name==='AlphaFold3');
  ours.forEach((r,i)=>reveal(c,clamp((p-.12*i)*3),()=>{
    t(c,(af[i].seconds/r.seconds).toFixed(1)+'×',xm(i),ym(Math.sqrt(r.seconds*af[i].seconds)),12,C.structure,'center',600);
  }));
  t(c,'RTX A6000 · denoising + validity',x+w/2,y+h-15,12,C.muted,'center');
}

function compactHigher(c,box,p){
  const[x,y,w,h]=box,b=[x+60,y+83,w-79,h-163],lo=-.25,hi=1;
  const ym=v=>b[1]+b[3]*(1-(v-lo)/(hi-lo));
  compactLegend(c,[['ProTEUS',C.blue],['MULTI-evolve',palette['MULTI-evolve']]],x+15,y+22,w-30);
  compactAxes(c,b,[],[-.25,0,.5,1].map(v=>[(v-lo)/(hi-lo),v===-.25?'−0.25':v.toFixed(1)]),'','Test Spearman ρ');
  [1,2,3].forEach((k,i)=>{
    const rs=DATA.higher.filter(r=>r.k===k),center=b[0]+b[2]*(i+.5)/3;
    const bw=Math.min(28,(b[2]/3-12)/2);
    [['ours',C.blue],['baseline',palette['MULTI-evolve']]].forEach(([key,col],j)=>{
      const vals=rs.map(r=>r[key]),mean=vals.reduce((a,v)=>a+v,0)/vals.length;
      const sem=Math.sqrt(vals.reduce((s,v)=>s+(v-mean)**2,0)/(vals.length-1)/vals.length);
      const xx=center+(j?3:-bw-3),q=clamp(p*1.5-i*.13-j*.035);
      c.save();c.globalAlpha*=.2;c.fillStyle=col;
      c.fillRect(xx,ym(mean*q),bw,ym(0)-ym(mean*q));c.restore();
      reveal(c,q,()=>{
        path(c,[[xx+bw/2,ym(mean-sem)],[xx+bw/2,ym(mean+sem)]],col,1.4);
        [mean-sem,mean+sem].forEach(v=>path(c,[[xx+bw/2-4,ym(v)],[xx+bw/2+4,ym(v)]],col,1.2));
        vals.forEach((v,n)=>dot(c,xx+4+((n*11)%rs.length)/(rs.length-1)*(bw-8),ym(v),2.2,col));
        path(c,[[xx,ym(mean)],[xx+bw,ym(mean)]],col,2.3);
      });
    });
    t(c,'K = '+k,center,b[1]+b[3]+20,13,C.ink,'center');
    t(c,'n = '+rs.length,center,b[1]+b[3]+40,12,C.muted,'center');
  });
  t(c,'Train on ≤K mutations; test on >K',x+w/2,y+h-15,13,C.muted,'center');
}

function compactEvolution(c,box,p){
  const[x,y,w,h]=box,b=[x+60,y+103,w-83,h-178],lo=.66,hi=1;
  const names=['ProTEUS','EVOLVEpro','Additive','MULTI-evolve'];
  const xm=v=>b[0]+b[2]*(v-1)/4,ym=v=>b[1]+b[3]*(1-(v-lo)/(hi-lo));
  compactLegend(c,names.map(n=>[n,palette[n]]),x+15,y+19,w-30);
  compactAxes(c,b,[1,2,3,4,5].map(v=>[(v-1)/4,String(v)]),[.7,.8,.9,1].map(v=>[(v-lo)/(hi-lo),v.toFixed(1)]),'Model-guided round','Normalized best fitness');
  names.slice().reverse().forEach(name=>{
    const rows=DATA.multi.filter(r=>r.name===name),col=palette[name];
    const pts=rows.map(r=>[xm(r.x),ym(r.y)]);
    band(c,b,rows,xm,ym,p,col);
    lineProgress(c,pts,p,col,name==='ProTEUS'?2.6:1.7);
    points(c,pts,p,col,name==='ProTEUS'?4:3);
  });
  t(c,'17 datasets · mean ± SEM',x+w/2,y+h-15,12,C.muted,'center');
}

const compactCharts={Fig2C:compactScaling,Fig2E:compactZero,ZeroAssays:compactAssays,SelectionQuality:compactQualityPanel,Fig2F_selection:compactSelection,Fig2G:compactFine,Fig3D:compactRuntime,Fig3F:compactHigher,Fig3G:compactEvolution};
function drawChart(c,name,box,p=1){p=clamp(p);const[x,y,w,h]=box;c.save();
  if(w<600&&compactCharts[name]){
    compactCharts[name](c,box,p);c.restore();return{x,y,w,h};
  }
  if(name==='Fig2C')scaling(c,box,p);
  else if(name==='Fig2E')zero(c,box,p);
  else if(name==='ZeroAssays')fullZero(c,box,p);
  else if(name==='SelectionQuality')selectionPanel(c,box,p);
  else if(name==='Fig2G')fine(c,box,p);
  else if(name==='Fig2F_selection')seriesChart(c,box,p,DATA.selection,{xmin:5,xmax:100,ymin:.55,ymax:.85,xticks:[5,10,24,50,100],yticks:[.6,.7,.8],logx:true,percent:true,breakY:true,xlabel:'Selected candidates · logarithmic scale',ylabel:'Fraction at or above assay mean',note:'13 assays · mean ± SEM'});
  else if(name==='Fig2H_fitness')seriesChart(c,box,p,DATA.single,{xmin:1,xmax:5,ymin:.15,ymax:.85,xticks:[1,2,3,4,5],yticks:[.25,.5,.75],xlabel:'Model-guided round',ylabel:'Normalized best fitness',note:'10 datasets · mean ± SEM'});
  else if(name==='Fig3C_CASP15')quality(c,box,p);
  else if(name==='Fig3C_legend')qualityLegend(c,box);
  else if(name==='Fig3D')runtime(c,box,p);
  else if(name==='Fig3F')higher(c,box,p);
  else if(name==='Fig3G')seriesChart(c,box,p,DATA.multi,{xmin:1,xmax:5,ymin:.66,ymax:1,xticks:[1,2,3,4,5],yticks:[.7,.8,.9,1],decimals:1,xlabel:'Model-guided round',ylabel:'Normalized best fitness',note:'17 datasets · mean ± SEM'});
  else throw new Error('Unknown computational chart: '+name);
  c.restore();return{x,y,w,h};
}
module.exports={drawChart,layoutForWidth,background:C.paper};

})();
return module.exports;})();
