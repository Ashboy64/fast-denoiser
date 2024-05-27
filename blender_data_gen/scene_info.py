from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SceneInfo:
    name: str
    scene_file_path: str
    view_layer_name: str
    camera_name: str
    aux_feature_image_name: str
    reference_camera_pos: List[np.ndarray[np.float32]]


BARBERSHOP_INFO = SceneInfo(
    name="barbershop",
    scene_file_path="./scenes/barbershop_interior.blend",
    view_layer_name="RenderLayer",
    camera_name="Camera",
    aux_feature_image_name="Image0001.png",
    reference_camera_pos=[
        np.array(
            [
                2.6868033409118652,
                7.4599690437316895,
                1.913818120956421,
                1.204020619392395,
                0.012979624792933464,
                7.082852840423584,
            ]
        ),
        np.array(
            [
                2.6868033409118652,
                7.4599690437316895,
                1.913818120956421,
                1.198793888092041,
                0.012982228770852089,
                5.640353202819824,
            ]
        ),
        np.array(
            [
                2.492485523223877,
                6.066747188568115,
                1.375929355621338,
                1.2799910306930542,
                0.012976129539310932,
                7.439016342163086,
            ]
        ),
        np.array(
            [
                1.7616899013519287,
                4.508598327636719,
                1.705711007118225,
                1.0915107727050781,
                0.012985683046281338,
                11.010051727294922,
            ]
        ),
        np.array(
            [
                1.7616899013519287,
                4.508598327636719,
                1.705711007118225,
                1.2957290410995483,
                0.012986859306693077,
                12.43690299987793,
            ]
        ),
        np.array(
            [
                1.0130856037139893,
                2.820591926574707,
                1.220571517944336,
                1.2748388051986694,
                0.013001743704080582,
                17.259376525878906,
            ]
        ),
        np.array(
            [
                1.879535436630249,
                4.087125301361084,
                1.220571517944336,
                1.3874186277389526,
                0.013001963496208191,
                16.170297622680664,
            ]
        ),
        np.array(
            [
                2.380704164505005,
                3.124859094619751,
                1.2776596546173096,
                1.4764376878738403,
                0.012999217957258224,
                15.141534805297852,
            ]
        ),
        np.array(
            [
                1.6977511644363403,
                2.564833164215088,
                1.5888128280639648,
                1.541951060295105,
                0.012975407764315605,
                13.979585647583008,
            ]
        ),
        np.array(
            [
                1.9652340412139893,
                3.911712408065796,
                1.5888128280639648,
                1.4372427463531494,
                0.01296931505203247,
                13.85137939453125,
            ]
        ),
        np.array(
            [
                1.2661653757095337,
                6.813700199127197,
                1.387590765953064,
                1.4660855531692505,
                0.012965488247573376,
                18.770719528198242,
            ]
        ),
        np.array(
            [
                2.1757240295410156,
                6.700595855712891,
                2.181436777114868,
                0.9844205379486084,
                0.01297683734446764,
                17.83344841003418,
            ]
        ),
        np.array(
            [
                3.03310227394104,
                7.719754695892334,
                2.1336629390716553,
                1.0657378435134888,
                0.012990506365895271,
                18.80478858947754,
            ]
        ),
    ],
)


ITALIAN_FLAT_INFO = SceneInfo(
    name="archiviz",
    scene_file_path="./scenes/flat-archiviz.blend",
    view_layer_name="View Layer",
    camera_name="Cam Sofa Back",
    aux_feature_image_name="Image0003.png",
    reference_camera_pos=[
        np.array(
            [
                -0.4624236524105072,
                -1.6774489879608154,
                1.204875111579895,
                1.4739410877227783,
                8.245119147431978e-07,
                -0.5849612355232239,
            ]
        ),
        np.array(
            [
                -1.9797773361206055,
                0.3692364990711212,
                2.629070520401001,
                1.178130865097046,
                5.470884389069397e-06,
                -1.1190485954284668,
            ]
        ),
        np.array(
            [
                4.104016304016113,
                4.793098449707031,
                2.957864999771118,
                1.1991198062896729,
                7.778563485771883e-06,
                -3.794731378555298,
            ]
        ),
        np.array(
            [
                5.8320536613464355,
                4.954078197479248,
                4.120606899261475,
                0.8195439577102661,
                1.1101590189355193e-06,
                -5.287040710449219,
            ]
        ),
        np.array(
            [
                1.207487940788269,
                9.725935935974121,
                4.0774760246276855,
                0.9687954187393188,
                7.838300916773733e-06,
                -8.504682540893555,
            ]
        ),
        np.array(
            [
                3.720721960067749,
                7.914767742156982,
                3.208054780960083,
                1.1860934495925903,
                7.687697689107154e-06,
                -10.114784240722656,
            ]
        ),
        np.array(
            [
                2.880504846572876,
                9.469766616821289,
                2.795560836791992,
                1.2908436059951782,
                4.429605723998975e-06,
                -15.7201509475708,
            ]
        ),
        np.array(
            [
                0.3750305473804474,
                4.244366645812988,
                2.462674140930176,
                1.3877344131469727,
                7.197553713922389e-06,
                -14.887760162353516,
            ]
        ),
        np.array(
            [
                3.345412015914917,
                3.6068146228790283,
                3.7345757484436035,
                1.0265339612960815,
                -7.817018286004895e-07,
                -25.266075134277344,
            ]
        ),
        np.array(
            [
                1.9555848836898804,
                -0.7249310612678528,
                1.5683075189590454,
                1.4743711948394775,
                8.448045264231041e-06,
                -25.100831985473633,
            ]
        ),
    ],
)

CLASSROOM_INFO = SceneInfo(
    name="classroom",
    scene_file_path="./scenes/classroom/classroom.blend",
    view_layer_name="interior",
    camera_name="Camera",
    aux_feature_image_name="Image0001.png",
    reference_camera_pos=[
        np.array(
            [
                2.576395273208618,
                -4.465750694274902,
                1.094475507736206,
                1.5819953680038452,
                3.25860833072511e-06,
                0.25481462478637695,
            ]
        ),
        np.array(
            [
                2.8309147357940674,
                -3.001325845718384,
                1.7191882133483887,
                1.4511008262634277,
                9.57443717197748e-07,
                1.6475881338119507,
            ]
        ),
        np.array(
            [
                0.8357277512550354,
                -4.250850677490234,
                2.0564661026000977,
                1.5977354049682617,
                1.5161617739067879e-05,
                -0.7609918713569641,
            ]
        ),
        np.array(
            [
                -0.4849654734134674,
                -2.8069472312927246,
                0.8680052161216736,
                1.2443486452102661,
                6.792260592192179e-06,
                2.493175745010376,
            ]
        ),
        np.array(
            [
                -0.09100060164928436,
                1.1138038635253906,
                1.5776495933532715,
                1.498311161994934,
                -5.939983111602487e-06,
                1.6266303062438965,
            ]
        ),
        np.array(
            [
                -0.3006440997123718,
                0.6313178539276123,
                1.6596485376358032,
                1.273171067237854,
                -7.224236014735652e-06,
                0.3254891037940979,
            ]
        ),
        np.array(
            [
                0.8298668265342712,
                0.5137124061584473,
                1.731061339378357,
                1.5821146965026855,
                -6.1487144193961285e-06,
                -0.006999018602073193,
            ]
        ),
        np.array(
            [
                1.1274628639221191,
                1.3732091188430786,
                1.8063400983810425,
                1.3229436874389648,
                -2.946389713542885e-06,
                -0.7662237286567688,
            ]
        ),
        np.array(
            [
                2.940042018890381,
                2.120189666748047,
                1.909437894821167,
                0.6736802458763123,
                1.2430627066351008e-05,
                -3.913100004196167,
            ]
        ),
        np.array(
            [
                -1.1751595735549927,
                -4.300800323486328,
                2.0824930667877197,
                1.2182408571243286,
                2.0577628674800508e-05,
                -0.551630437374115,
            ]
        ),
        np.array(
            [
                0.6757554411888123,
                -1.2991529703140259,
                1.4262173175811768,
                1.3255845308303833,
                1.758587131917011e-05,
                -3.766547679901123,
            ]
        ),
    ],
)

SCENE_INFO = {
    "barbershop": BARBERSHOP_INFO,
    "italian_flat": ITALIAN_FLAT_INFO,
    "classroom": CLASSROOM_INFO,
}
