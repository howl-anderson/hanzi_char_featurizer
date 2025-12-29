"""拼音部分特征提取示例"""

from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts


def main():
    pp = PinYinParts()

    # 读取测试数据
    with open("data.txt", "rt") as fd:
        for line in fd:
            text = line.strip()
            if not text:
                continue

            print(f"\n输入: {text}")
            print("-" * 40)

            # 提取特征 - 字典格式
            result = pp.extract(text)
            print(f"extract(): {result}")

            # 提取特征 - NumPy 格式
            result_numpy = pp.extract(text, as_numpy=True)
            print(f"extract(as_numpy=True): {result_numpy}")

    # 获取词汇表
    print("\n" + "=" * 40)
    print("词汇表:")
    print(pp.vocabulary)


if __name__ == "__main__":
    main()
