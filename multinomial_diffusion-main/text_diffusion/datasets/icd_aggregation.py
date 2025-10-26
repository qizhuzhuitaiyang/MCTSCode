#!/usr/bin/env python
# coding: utf-8

"""
ICD 编码聚合工具
利用 ICD 编码的原生层级结构进行聚合
"""


def aggregate_diagnosis_code(code):
    """
    将 ICD 诊断码聚合到 3 位类目
    
    规则：
    - ICD-9-CM E 编码（外因）：取前 4 位（类目是 4 位）
    - ICD-9-CM V 编码（补充分类）：取前 4 位（类目是 4 位）
    - ICD-9-CM 数字编码：取前 3 位
    - ICD-10-CM：取前 3 位
    
    Args:
        code: 原始 ICD 诊断码（如 '5723', 'E785', 'G3183'）
    
    Returns:
        聚合后的类目（如 '572', 'E785', 'G31'）
    
    Examples:
        >>> aggregate_diagnosis_code('5723')
        '572'
        >>> aggregate_diagnosis_code('E785')
        'E785'
        >>> aggregate_diagnosis_code('V1582')
        'V158'
        >>> aggregate_diagnosis_code('G3183')
        'G31'
        >>> aggregate_diagnosis_code('E1165')  # MIMIC 中无小数点
        'E11'
    """
    if not code:
        return ''
    
    code = str(code).strip()
    
    # 特殊处理：区分 ICD-9 和 ICD-10 的 E 编码
    if code.startswith('E'):
        # ICD-9 E编码: E7xx-E9xx（4位类目）
        # ICD-10 E编码: E00-E89（3位类目）
        # 判断：如果第2位是 7/8/9，是 ICD-9，取前4位
        if len(code) >= 2 and code[1] in ['7', '8', '9']:
            return code[:4] if len(code) >= 4 else code  # E785 → E785, E8497 → E849
        # 否则是 ICD-10，取前3位
        else:
            return code[:3]  # E1165 → E11
    
    # 特殊处理：ICD-9-CM 的 V 编码（补充分类，如 V01-V91）
    elif code.startswith('V'):
        # V 编码的类目是 4 位
        if len(code) >= 4:
            return code[:4]
        else:
            return code
    
    # 通用规则：取前 3 位（适用于 ICD-9-CM 数字编码和 ICD-10-CM）
    else:
        if len(code) >= 3:
            return code[:3]
        else:
            return code


def aggregate_procedure_code(code):
    """
    将 ICD 手术码聚合到 2 位大类
    
    规则：
    - ICD-9-PCS：取前 2 位（章节级别）
    - ICD-10-PCS：取前 2 位（章节 + 身体系统）
    
    Args:
        code: 原始 ICD 手术码（如 '5491', '0QS734Z'）
    
    Returns:
        聚合后的大类（如 '54', '0Q'）
    
    Examples:
        >>> aggregate_procedure_code('5491')
        '54'
        >>> aggregate_procedure_code('0QS734Z')
        '0Q'
        >>> aggregate_procedure_code('3995')
        '39'
    """
    if not code:
        return ''
    
    code = str(code).strip()
    
    if len(code) >= 2:
        return code[:2]
    else:
        return code


def test_aggregation():
    """测试聚合函数"""
    print("=== 测试 ICD 编码聚合 ===\n")
    
    # 测试诊断编码
    print("1. 诊断编码聚合测试:")
    test_diagnoses = [
        ('5723', '572', 'ICD-9 数字编码'),
        ('25000', '250', 'ICD-9 数字编码'),
        ('4019', '401', 'ICD-9 数字编码'),
        ('E785', 'E785', 'ICD-9 E编码'),
        ('E8497', 'E849', 'ICD-9 E编码'),
        ('V1582', 'V158', 'ICD-9 V编码'),
        ('V4581', 'V458', 'ICD-9 V编码'),
        ('G3183', 'G31', 'ICD-10'),
        ('E1165', 'E11', 'ICD-10'),
        ('I10', 'I10', 'ICD-10 (已是3位)'),
    ]
    
    for code, expected, desc in test_diagnoses:
        result = aggregate_diagnosis_code(code)
        status = '✓' if result == expected else '✗'
        print(f"  {status} {code:10s} → {result:6s} (期望: {expected:6s}) [{desc}]")
    
    # 测试手术编码
    print("\n2. 手术编码聚合测试:")
    test_procedures = [
        ('5491', '54', 'ICD-9-PCS'),
        ('3995', '39', 'ICD-9-PCS'),
        ('8938', '89', 'ICD-9-PCS'),
        ('0QS734Z', '0Q', 'ICD-10-PCS'),
        ('02HV33Z', '02', 'ICD-10-PCS'),
        ('0TTB4ZZ', '0T', 'ICD-10-PCS'),
    ]
    
    for code, expected, desc in test_procedures:
        result = aggregate_procedure_code(code)
        status = '✓' if result == expected else '✗'
        print(f"  {status} {code:10s} → {result:6s} (期望: {expected:6s}) [{desc}]")
    
    print("\n测试完成！")


if __name__ == '__main__':
    test_aggregation()

