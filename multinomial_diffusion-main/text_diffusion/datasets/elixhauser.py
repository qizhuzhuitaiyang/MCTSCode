#!/usr/bin/env python
# coding: utf-8

"""
Elixhauser 并存病指标提取
基于 ICD-9-CM 和 ICD-10-CM 编码推导 31 项并存病指标
"""

import torch


# Elixhauser 31 项并存病的 ICD-9-CM 和 ICD-10-CM 前缀规则
# 参考: https://www.hcup-us.ahrq.gov/toolssoftware/comorbidity/comorbidity.jsp

ELIXHAUSER_ICD9_RULES = {
    'CHF': ['398', '402', '404', '428'],  # 充血性心力衰竭
    'ARRHYTHMIA': ['426', '427', 'V450', 'V533'],  # 心律失常
    'VALVE': ['093', '394', '395', '396', '397', '424', 'V422', 'V433'],  # 瓣膜病
    'PULM_CIRC': ['415', '416', '417'],  # 肺循环疾病
    'PVD': ['440', '441', '442', '443', '444', '447', '557', 'V434'],  # 外周血管疾病
    'HTN_UNCOMPLICATED': ['401'],  # 高血压（无并发症）
    'HTN_COMPLICATED': ['402', '403', '404', '405'],  # 高血压（有并发症）
    'PARALYSIS': ['342', '343', '344'],  # 瘫痪
    'NEURO_OTHER': ['330', '331', '332', '333', '334', '335', '336', '340', '341', '345'],  # 其他神经系统疾病
    'CHRONIC_PULM': ['490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505'],  # 慢性肺病
    'DM_UNCOMPLICATED': ['250'],  # 糖尿病（无并发症）- 需排除 2501-2503
    'DM_COMPLICATED': ['250'],  # 糖尿病（有并发症）- 仅 2501-2503
    'HYPOTHYROID': ['243', '244'],  # 甲状腺功能减退
    'RENAL_FAILURE': ['585', '586', 'V420', 'V451', 'V56'],  # 肾衰竭
    'LIVER_DISEASE': ['070', '456', '570', '571', '572', '573', 'V427'],  # 肝病
    'PUD': ['531', '532', '533', '534'],  # 消化性溃疡病
    'HIV': ['042', '043', '044'],  # HIV/AIDS
    'LYMPHOMA': ['200', '201', '202', '203'],  # 淋巴瘤
    'METASTATIC_CANCER': ['196', '197', '198', '199'],  # 转移性癌
    'SOLID_TUMOR': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                     '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                     '160', '161', '162', '163', '164', '165', '170', '171', '172', '174',
                     '175', '176', '179', '180', '181', '182', '183', '184', '185', '186',
                     '187', '188', '189', '190', '191', '192', '193', '194', '195'],  # 实体瘤
    'RHEUMATOID': ['701', '710', '714', '720', '725'],  # 类风湿性关节炎/胶原血管病
    'COAGULOPATHY': ['286', '287'],  # 凝血功能障碍
    'OBESITY': ['278'],  # 肥胖
    'WEIGHT_LOSS': ['260', '261', '262', '263'],  # 体重减轻
    'FLUID_ELECTROLYTE': ['276'],  # 液体和电解质紊乱
    'BLOOD_LOSS_ANEMIA': ['280'],  # 失血性贫血
    'DEFICIENCY_ANEMIA': ['281'],  # 缺乏性贫血
    'ALCOHOL_ABUSE': ['291', '303', '305'],  # 酒精滥用
    'DRUG_ABUSE': ['292', '304'],  # 药物滥用
    'PSYCHOSES': ['295', '296', '297', '298'],  # 精神病
    'DEPRESSION': ['300', '309', '311'],  # 抑郁症
}

ELIXHAUSER_ICD10_RULES = {
    'CHF': ['I09', 'I11', 'I13', 'I50'],
    'ARRHYTHMIA': ['I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'R00', 'T82', 'Z45', 'Z95'],
    'VALVE': ['A52', 'I05', 'I06', 'I07', 'I08', 'I09', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'Q23', 'Z95'],
    'PULM_CIRC': ['I26', 'I27', 'I28'],
    'PVD': ['I70', 'I71', 'I73', 'I77', 'I79', 'K55', 'Z95'],
    'HTN_UNCOMPLICATED': ['I10'],
    'HTN_COMPLICATED': ['I11', 'I12', 'I13', 'I15'],
    'PARALYSIS': ['G04', 'G11', 'G80', 'G81', 'G82', 'G83'],
    'NEURO_OTHER': ['G10', 'G20', 'G21', 'G22', 'G25', 'G30', 'G31', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41'],
    'CHRONIC_PULM': ['I27', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67'],
    'DM_UNCOMPLICATED': ['E10', 'E11', 'E12', 'E13', 'E14'],  # 需排除带并发症的
    'DM_COMPLICATED': ['E10', 'E11', 'E12', 'E13', 'E14'],  # 仅带并发症的
    'HYPOTHYROID': ['E00', 'E01', 'E02', 'E03', 'E89'],
    'RENAL_FAILURE': ['I12', 'I13', 'N18', 'N19', 'N25', 'Z49', 'Z94', 'Z99'],
    'LIVER_DISEASE': ['B18', 'I85', 'I86', 'I98', 'K70', 'K71', 'K72', 'K73', 'K74', 'K76', 'Z94'],
    'PUD': ['K25', 'K26', 'K27', 'K28'],
    'HIV': ['B20', 'B21', 'B22', 'B24'],
    'LYMPHOMA': ['C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C96'],
    'METASTATIC_CANCER': ['C77', 'C78', 'C79', 'C80'],
    'SOLID_TUMOR': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
                     'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
                     'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32',
                     'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46',
                     'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56',
                     'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67',
                     'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C97'],
    'RHEUMATOID': ['L94', 'M05', 'M06', 'M08', 'M12', 'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M45'],
    'COAGULOPATHY': ['D65', 'D66', 'D67', 'D68', 'D69'],
    'OBESITY': ['E66'],
    'WEIGHT_LOSS': ['E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46'],
    'FLUID_ELECTROLYTE': ['E22', 'E86', 'E87'],
    'BLOOD_LOSS_ANEMIA': ['D50'],
    'DEFICIENCY_ANEMIA': ['D51', 'D52', 'D53'],
    'ALCOHOL_ABUSE': ['F10', 'E52', 'G62', 'I42', 'K29', 'K70', 'T51', 'Z50', 'Z71', 'Z72'],
    'DRUG_ABUSE': ['F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19', 'Z71', 'Z72'],
    'PSYCHOSES': ['F20', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F39'],
    'DEPRESSION': ['F32', 'F33', 'F34', 'F41', 'F43'],
}

# Elixhauser 31 项并存病名称（按顺序）
ELIXHAUSER_NAMES = [
    'CHF', 'ARRHYTHMIA', 'VALVE', 'PULM_CIRC', 'PVD',
    'HTN_UNCOMPLICATED', 'HTN_COMPLICATED', 'PARALYSIS', 'NEURO_OTHER', 'CHRONIC_PULM',
    'DM_UNCOMPLICATED', 'DM_COMPLICATED', 'HYPOTHYROID', 'RENAL_FAILURE', 'LIVER_DISEASE',
    'PUD', 'HIV', 'LYMPHOMA', 'METASTATIC_CANCER', 'SOLID_TUMOR',
    'RHEUMATOID', 'COAGULOPATHY', 'OBESITY', 'WEIGHT_LOSS', 'FLUID_ELECTROLYTE',
    'BLOOD_LOSS_ANEMIA', 'DEFICIENCY_ANEMIA', 'ALCOHOL_ABUSE', 'DRUG_ABUSE', 'PSYCHOSES',
    'DEPRESSION'
]


def extract_elixhauser_comorbidities(icd_codes):
    """
    从 ICD 编码列表中提取 Elixhauser 31 项并存病指标
    
    Args:
        icd_codes: ICD 编码列表（可混合 ICD-9 和 ICD-10）
    
    Returns:
        elixhauser_vector: Tensor (31,)，0/1 二值向量
    """
    elixhauser = torch.zeros(31)
    
    if not icd_codes:
        return elixhauser
    
    # 将所有编码转为字符串并去重
    icd_codes = [str(code).strip() for code in icd_codes if code]
    icd_codes_set = set(icd_codes)
    
    # 遍历 31 项并存病
    for idx, name in enumerate(ELIXHAUSER_NAMES):
        # 检查 ICD-9-CM 规则
        icd9_prefixes = ELIXHAUSER_ICD9_RULES.get(name, [])
        for code in icd_codes_set:
            for prefix in icd9_prefixes:
                if code.startswith(prefix):
                    # 特殊处理：糖尿病需区分有无并发症
                    if name == 'DM_UNCOMPLICATED' and prefix == '250':
                        # 排除 2501, 2502, 2503（有并发症）
                        if not (code.startswith('2501') or code.startswith('2502') or code.startswith('2503')):
                            elixhauser[idx] = 1
                            break
                    elif name == 'DM_COMPLICATED' and prefix == '250':
                        # 仅包含 2501, 2502, 2503
                        if code.startswith('2501') or code.startswith('2502') or code.startswith('2503'):
                            elixhauser[idx] = 1
                            break
                    else:
                        elixhauser[idx] = 1
                        break
            if elixhauser[idx] == 1:
                break
        
        # 如果 ICD-9 未匹配，检查 ICD-10-CM 规则
        if elixhauser[idx] == 0:
            icd10_prefixes = ELIXHAUSER_ICD10_RULES.get(name, [])
            for code in icd_codes_set:
                for prefix in icd10_prefixes:
                    if code.startswith(prefix):
                        # 特殊处理：糖尿病需区分有无并发症
                        if name == 'DM_UNCOMPLICATED' and prefix in ['E10', 'E11', 'E12', 'E13', 'E14']:
                            # 排除带 .2-.9 的（有并发症）
                            if len(code) >= 4 and code[3] in ['2', '3', '4', '5', '6', '7', '8', '9']:
                                continue
                            else:
                                elixhauser[idx] = 1
                                break
                        elif name == 'DM_COMPLICATED' and prefix in ['E10', 'E11', 'E12', 'E13', 'E14']:
                            # 仅包含带 .2-.9 的
                            if len(code) >= 4 and code[3] in ['2', '3', '4', '5', '6', '7', '8', '9']:
                                elixhauser[idx] = 1
                                break
                        else:
                            elixhauser[idx] = 1
                            break
                if elixhauser[idx] == 1:
                    break
    
    return elixhauser


def test_elixhauser():
    """测试 Elixhauser 提取"""
    print("=== 测试 Elixhauser 并存病提取 ===\n")
    
    # 测试用例
    test_cases = [
        {
            'name': '高血压 + 糖尿病（无并发症）',
            'codes': ['4019', '25000'],
            'expected': ['HTN_UNCOMPLICATED', 'DM_UNCOMPLICATED']
        },
        {
            'name': '糖尿病（有并发症）+ 心力衰竭',
            'codes': ['25010', '428'],
            'expected': ['DM_COMPLICATED', 'CHF']
        },
        {
            'name': 'ICD-10: 2型糖尿病 + 高血压',
            'codes': ['E1165', 'I10'],
            'expected': ['DM_COMPLICATED', 'HTN_UNCOMPLICATED']
        },
        {
            'name': '慢性肺病 + 肾衰竭',
            'codes': ['496', '585'],
            'expected': ['CHRONIC_PULM', 'RENAL_FAILURE']
        },
    ]
    
    for test in test_cases:
        print(f"测试: {test['name']}")
        print(f"  输入编码: {test['codes']}")
        
        elixhauser = extract_elixhauser_comorbidities(test['codes'])
        
        # 找出被标记为 1 的并存病
        detected = [ELIXHAUSER_NAMES[i] for i in range(31) if elixhauser[i] == 1]
        
        print(f"  检测到: {detected}")
        print(f"  期望: {test['expected']}")
        
        # 检查是否匹配
        match = set(detected) == set(test['expected'])
        status = '✓' if match else '✗'
        print(f"  {status} {'通过' if match else '失败'}\n")
    
    print("测试完成！")


if __name__ == '__main__':
    test_elixhauser()

