#include <gtest/gtest.h>
#include "main.hpp"
#include "cr.hpp"




class Examples : public ::testing::Test {
public:
    struct TRIDIAG_SYSTEM *sys = nullptr;
    const static int n = 1024;

    real ans_array[n] = { 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, 16.5, 18, 19.5, 21, 22.5, 24, 25.5,
         27, 28.5, 30, 31.5, 33, 34.5, 36, 37.5, 39, 40.5, 42, 43.5, 45, 46.5, 48, 49.5, 51, 52.5, 54,
         55.5, 57, 58.5, 60, 61.5, 63, 64.5, 66, 67.5, 69, 70.5, 72, 73.5, 75, 76.5, 78, 79.5, 81,
         82.5, 84, 85.5, 87, 88.5, 90, 91.5, 93, 94.5, 96, 97.5, 99, 100.5, 102, 103.5, 105, 106.5,
         108, 109.5, 111, 112.5, 114, 115.5, 117, 118.5, 120, 121.5, 123, 124.5, 126, 127.5, 129, 130.5,
         132, 133.5, 135, 136.5, 138, 139.5, 141, 142.5, 144, 145.5, 147, 148.5, 150, 151.5, 153, 154.5,
         156, 157.5, 159, 160.5, 162, 163.5, 165, 166.5, 168, 169.5, 171, 172.5, 174, 175.5, 177, 178.5,
         180, 181.5, 183, 184.5, 186, 187.5, 189, 190.5, 192, 193.5, 195, 196.5, 198, 199.5, 201, 202.5,
         204, 205.5, 207, 208.5, 210, 211.5, 213, 214.5, 216, 217.5, 219, 220.5, 222, 223.5, 225, 226.5,
         228, 229.5, 231, 232.5, 234, 235.5, 237, 238.5, 240, 241.5, 243, 244.5, 246, 247.5, 249, 250.5,
         252, 253.5, 255, 256.5, 258, 259.5, 261, 262.5, 264, 265.5, 267, 268.5, 270, 271.5, 273, 274.5,
         276, 277.5, 279, 280.5, 282, 283.5, 285, 286.5, 288, 289.5, 291, 292.5, 294, 295.5, 297, 298.5,
         300, 301.5, 303, 304.5, 306, 307.5, 309, 310.5, 312, 313.5, 315, 316.5, 318, 319.5, 321, 322.5,
         324, 325.5, 327, 328.5, 330, 331.5, 333, 334.5, 336, 337.5, 339, 340.5, 342, 343.5, 345, 346.5,
         348, 349.5, 351, 352.5, 354, 355.5, 357, 358.5, 360, 361.5, 363, 364.5, 366, 367.5, 369, 370.5,
         372, 373.5, 375, 376.5, 378, 379.5, 381, 382.5, 384, 385.5, 387, 388.5, 390, 391.5, 393, 394.5,
         396, 397.5, 399, 400.5, 402, 403.5, 405, 406.5, 408, 409.5, 411, 412.5, 414, 415.5, 417, 418.5,
         420, 421.5, 423, 424.5, 426, 427.5, 429, 430.5, 432, 433.5, 435, 436.5, 438, 439.5, 441, 442.5,
         444, 445.5, 447, 448.5, 450, 451.5, 453, 454.5, 456, 457.5, 459, 460.5, 462, 463.5, 465, 466.5,
         468, 469.5, 471, 472.5, 474, 475.5, 477, 478.5, 480, 481.5, 483, 484.5, 486, 487.5, 489, 490.5,
         492, 493.5, 495, 496.5, 498, 499.5, 501, 502.5, 504, 505.5, 507, 508.5, 510, 511.5, 513, 514.5,
         516, 517.5, 519, 520.5, 522, 523.5, 525, 526.5, 528, 529.5, 531, 532.5, 534, 535.5, 537, 538.5,
         540, 541.5, 543, 544.5, 546, 547.5, 549, 550.5, 552, 553.5, 555, 556.5, 558, 559.5, 561, 562.5,
         564, 565.5, 567, 568.5, 570, 571.5, 573, 574.5, 576, 577.5, 579, 580.5, 582, 583.5, 585, 586.5,
         588, 589.5, 591, 592.5, 594, 595.5, 597, 598.5, 600, 601.5, 603, 604.5, 606, 607.5, 609, 610.5,
         612, 613.5, 615, 616.5, 618, 619.5, 621, 622.5, 624, 625.5, 627, 628.5, 630, 631.5, 633, 634.5,
         636, 637.5, 639, 640.5, 642, 643.5, 645, 646.5, 648, 649.5, 651, 652.5, 654, 655.5, 657, 658.5,
         660, 661.5, 663, 664.5, 666, 667.5, 669, 670.5, 672, 673.5, 675, 676.5, 678, 679.5, 681, 682.5,
         684, 685.5, 687, 688.5, 690, 691.5, 693, 694.5, 696, 697.5, 699, 700.5, 702, 703.5, 705, 706.5,
         708, 709.5, 711, 712.5, 714, 715.5, 717, 718.5, 720, 721.5, 723, 724.5, 726, 727.5, 729, 730.5,
         732, 733.5, 735, 736.5, 738, 739.5, 741, 742.5, 744, 745.5, 747, 748.5, 750, 751.5, 753, 754.5,
         756, 757.5, 759, 760.5, 762, 763.5, 765, 766.5, 768, 769.5, 771, 772.5, 774, 775.5, 777, 778.5,
         780, 781.5, 783, 784.5, 786, 787.5, 789, 790.5, 792, 793.5, 795, 796.5, 798, 799.5, 801, 802.5,
         804, 805.5, 807, 808.5, 810, 811.5, 813, 814.5, 816, 817.5, 819, 820.5, 822, 823.5, 825, 826.5,
         828, 829.5, 831, 832.5, 834, 835.5, 837, 838.5, 840, 841.5, 843, 844.5, 846, 847.5, 849, 850.5,
         852, 853.5, 855, 856.5, 858, 859.5, 861, 862.5, 864, 865.5, 867, 868.5, 870, 871.5, 873, 874.5,
         876, 877.5, 879, 880.5, 882, 883.5, 885, 886.5, 888, 889.5, 891, 892.5, 894, 895.5, 897, 898.5,
         900, 901.5, 903, 904.5, 906, 907.5, 909, 910.5, 912, 913.5, 915, 916.5, 918, 919.5, 921, 922.5,
         924, 925.5, 927, 928.5, 930, 931.5, 933, 934.5, 936, 937.5, 939, 940.5, 942, 943.5, 945, 946.5,
         948, 949.5, 951, 952.5, 954, 955.5, 957, 958.5, 960, 961.5, 963, 964.5, 966, 967.5, 969, 970.5,
         972, 973.5, 975, 976.5, 978, 979.5, 981, 982.5, 984, 985.5, 987, 988.5, 990, 991.5, 993, 994.5,
         996, 997.5, 999, 1000.5, 1002, 1003.5, 1005, 1006.5, 1008, 1009.5, 1011, 1012.5, 1014, 1015.5,
         1017, 1018.5, 1020, 1021.5, 1023, 1024.5, 1026, 1027.5, 1029, 1030.5, 1032, 1033.5, 1035, 1036.5,
         1038, 1039.5, 1041, 1042.5, 1044, 1045.5, 1047, 1048.5, 1050, 1051.5, 1053, 1054.5, 1056, 1057.5,
         1059, 1060.5, 1062, 1063.5, 1065, 1066.5, 1068, 1069.5, 1071, 1072.5, 1074, 1075.5, 1077, 1078.5,
         1080, 1081.5, 1083, 1084.5, 1086, 1087.5, 1089, 1090.5, 1092, 1093.5, 1095, 1096.5, 1098, 1099.5,
         1101, 1102.5, 1104, 1105.5, 1107, 1108.5, 1110, 1111.5, 1113, 1114.5, 1116, 1117.5, 1119, 1120.5,
         1122, 1123.5, 1125, 1126.5, 1128, 1129.5, 1131, 1132.5, 1134, 1135.5, 1137, 1138.5, 1140, 1141.5,
         1143, 1144.5, 1146, 1147.5, 1149, 1150.5, 1152, 1153.5, 1155, 1156.5, 1158, 1159.5, 1161, 1162.5,
         1164, 1165.5, 1167, 1168.5, 1170, 1171.5, 1173, 1174.5, 1176, 1177.5, 1179, 1180.5, 1182, 1183.5,
         1185, 1186.5, 1188, 1189.5, 1191, 1192.5, 1194, 1195.5, 1197, 1198.5, 1200, 1201.5, 1203, 1204.5,
         1206, 1207.5, 1209, 1210.5, 1212, 1213.5, 1215, 1216.5, 1218, 1219.5, 1221, 1222.5, 1224, 1225.5,
         1227, 1228.5, 1230, 1231.5, 1233, 1234.5, 1236, 1237.5, 1239, 1240.5, 1242, 1243.5, 1245, 1246.5,
         1248, 1249.5, 1251, 1252.5, 1254, 1255.5, 1257, 1258.5, 1260, 1261.5, 1263, 1264.5, 1266, 1267.5,
         1269, 1270.5, 1272, 1273.5, 1275, 1276.5, 1278, 1279.5, 1281, 1282.5, 1284, 1285.5, 1287, 1288.5,
         1290, 1291.5, 1293, 1294.5, 1296, 1297.5, 1299, 1300.5, 1302, 1303.5, 1305, 1306.5, 1308, 1309.5,
         1311, 1312.5, 1314, 1315.5, 1317, 1318.5, 1320, 1321.5, 1323, 1324.5, 1326, 1327.5, 1329, 1330.5,
         1332, 1333.5, 1335, 1336.5, 1338, 1339.5, 1341, 1342.5, 1344, 1345.5, 1347, 1348.5, 1350, 1351.5,
         1353, 1354.5, 1356, 1357.5, 1359, 1360.5, 1362, 1363.5, 1365, 1366.5, 1368, 1369.5, 1371, 1372.5,
         1374, 1375.5, 1377, 1378.5, 1380, 1381.5, 1383, 1384.5, 1386, 1387.5, 1389, 1390.5, 1392, 1393.5,
         1395, 1396.5, 1398, 1399.5, 1401, 1402.5, 1404, 1405.5, 1407, 1408.5, 1410, 1411.5, 1413, 1414.5,
         1416, 1417.5, 1419, 1420.5, 1422, 1423.5, 1425, 1426.5, 1428, 1429.5, 1431, 1432.5, 1434, 1435.5,
         1437, 1438.5, 1440, 1441.5, 1443, 1444.5, 1446, 1447.5, 1449, 1450.5, 1452, 1453.5, 1455, 1456.5,
         1458, 1459.5, 1461, 1462.5, 1464, 1465.5, 1467, 1468.5, 1470, 1471.5, 1473, 1474.5, 1476, 1477.5,
         1479, 1480.5, 1482, 1483.5, 1485, 1486.5, 1488, 1489.5, 1491, 1492.5, 1494, 1495.5, 1497, 1498.5,
         1500, 1501.5, 1503, 1504.5, 1506, 1507.5, 1509, 1510.5, 1512, 1513.5, 1515, 1516.5, 1518, 1519.5,
         1521, 1522.5, 1524, 1525.5, 1526.99, 1528.46, 1529.77, 1530.17, 1525.23, 1489.24, 1272.21};

protected:

    void SetUp() override {
        sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
        setup(sys, n);

        assign(sys);
    }

    void TearDown() override {
        clean(sys);
        free(sys);
    }

    void array_float_eq(real *expect, real *actual) {
        for (int i = 0; i < n; i++) {
            #ifdef _REAL_IS_DOUBLE_
                ASSERT_DOUBLE_EQ(expect[i], actual[i]) << "Expect " << expect[i] << " but got " << actual[i] << " at index " << i << "\n";
            #else
                ASSERT_FLOAT_EQ(expect[i], actual[i]) << "Expect " << expect[i] << " but got " << actual[i] << " at index " << i << "\n";
            #endif
        }
    }
};


TEST_F(Examples, CRTest) {
    #pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
    {
        CR cr(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        cr.solve();
        cr.get_ans(sys->diag);
        array_float_eq(ans_array, sys->diag);
    }
}

int setup(struct TRIDIAG_SYSTEM *sys, int n) {
    sys->a = (real *)malloc(n * sizeof(real));
    sys->diag = (real *)malloc(n * sizeof(real));
    sys->c = (real *)malloc(n * sizeof(real));
    sys->rhs = (real *)malloc(n * sizeof(real));
    sys->n = n;

    return sys_null_check(sys);
}

int assign(struct TRIDIAG_SYSTEM *sys) {
    int n = sys->n;
    for (int i = 0; i < n; i++) {
        sys->a[i] = -1.0/6.0;
        sys->c[i] = -1.0/6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = 1.0 * (i+1);
    }
    sys->a[0] = 0.0;
    sys->c[n-1] = 0.0;

    return 0;
}



int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        free(p);
    }

    sys->a = nullptr;
    sys->diag = nullptr;
    sys->c = nullptr;
    sys->rhs = nullptr;

    return 0;
}


bool sys_null_check(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}


