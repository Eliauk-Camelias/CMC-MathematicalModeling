"""
遗传算法模板使用示例
"""

from genetic_algorithm import GeneticAlgorithm


# 示例1: 简单的无约束优化问题
def example1():
    """最小化 f(x, y) = x^2 + y^2, 其中 x, y ∈ [-1, 1]"""
    print("=" * 60)
    print("示例1: 最小化 f(x, y) = x^2 + y^2 (无约束)")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        verbose=True
    )
    
    best, fitness = ga.run()
    print(f"\n最优解: x={best['x']:.6f}, y={best['y']:.6f}")
    print(f"最优适应度: {fitness:.6f}\n")


# 示例2: 使用约束函数（硬约束）
def example2():
    """最小化 f(x, y) = x^2 + y^2, 约束条件: x + y >= 0.5"""
    print("=" * 60)
    print("示例2: 带硬约束条件 x + y >= 0.5")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    def constraint_function(vars_dict):
        """返回True表示满足约束，False表示不满足"""
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x + y >= 0.5
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        constraint_function=constraint_function,  # 约束函数
        population_size=50,
        max_generations=100,
        verbose=True
    )
    
    best, fitness = ga.run()
    print(f"\n最优解: x={best['x']:.6f}, y={best['y']:.6f}")
    print(f"最优适应度: {fitness:.6f}")
    print(f"约束检查: x + y = {best['x'] + best['y']:.6f} >= 0.5 ✓\n")


# 示例3: 使用惩罚函数（软约束）
def example3():
    """使用惩罚函数处理约束条件"""
    print("=" * 60)
    print("示例3: 使用惩罚函数处理约束 x + y >= 0.5")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    def penalty_function(vars_dict, raw_fitness):
        """
        惩罚函数
        参数:
            vars_dict: 变量字典
            raw_fitness: 原始适应度值
        返回:
            惩罚后的适应度值
        """
        x = vars_dict["x"]
        y = vars_dict["y"]
        penalty = 0
        # 如果 x + y < 0.5，添加惩罚
        if x + y < 0.5:
            penalty = 100 * (0.5 - (x + y))**2
        return raw_fitness + penalty
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        penalty_function=penalty_function,  # 惩罚函数
        population_size=50,
        max_generations=100,
        verbose=True
    )
    
    best, fitness = ga.run()
    print(f"\n最优解: x={best['x']:.6f}, y={best['y']:.6f}")
    print(f"最优适应度: {fitness:.6f}")
    print(f"约束检查: x + y = {best['x'] + best['y']:.6f}\n")


# 示例4: 多变量复杂优化问题
def example4():
    """优化函数 f(x, y, z) = x^2 + y^2 + z^2 + x*y, 约束: x + y + z = 1"""
    print("=" * 60)
    print("示例4: 多变量优化 f(x,y,z) = x^2+y^2+z^2+x*y, 约束: x+y+z=1")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        z = vars_dict["z"]
        return x**2 + y**2 + z**2 + x * y
    
    def constraint_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        z = vars_dict["z"]
        # 允许一定误差
        return abs(x + y + z - 1) < 0.01
    
    ga = GeneticAlgorithm(
        variable_ranges={
            "x": (-2, 2),
            "y": (-2, 2),
            "z": (-2, 2)
        },
        fitness_function=fitness_function,
        constraint_function=constraint_function,
        population_size=100,
        max_generations=150,
        verbose=True
    )
    
    best, fitness = ga.run()
    print(f"\n最优解: x={best['x']:.6f}, y={best['y']:.6f}, z={best['z']:.6f}")
    print(f"最优适应度: {fitness:.6f}")
    print(f"约束检查: x + y + z = {best['x'] + best['y'] + best['z']:.6f} ≈ 1\n")


# 示例5: 结合约束函数和惩罚函数
def example5():
    """同时使用约束函数和惩罚函数"""
    print("=" * 60)
    print("示例5: 同时使用约束函数和惩罚函数")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    def constraint_function(vars_dict):
        """硬约束：x + y >= 0.5"""
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x + y >= 0.5
    
    def penalty_function(vars_dict, raw_fitness):
        """软惩罚：鼓励 x^2 + y^2 < 0.1"""
        x = vars_dict["x"]
        y = vars_dict["y"]
        penalty = 0
        if x**2 + y**2 > 0.1:
            penalty = 10 * (x**2 + y**2 - 0.1)
        return raw_fitness + penalty
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        constraint_function=constraint_function,
        penalty_function=penalty_function,
        population_size=50,
        max_generations=100,
        verbose=True
    )
    
    best, fitness = ga.run()
    print(f"\n最优解: x={best['x']:.6f}, y={best['y']:.6f}")
    print(f"最优适应度: {fitness:.6f}")
    print(f"约束检查: x + y = {best['x'] + best['y']:.6f} >= 0.5 ✓\n")


# 示例6: 使用自定义初始种群（种子）
def example6():
    """使用自定义初始种群 - 方式1：手动定义"""
    print("=" * 60)
    print("示例6a: 手动定义初始种群")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    # 方式1：手动定义（适用于少量种子）
    initial_seeds = [
        {"x": 0.1, "y": 0.1},
        {"x": -0.2, "y": 0.15},
        {"x": 0.05, "y": -0.1},
        {"x": -0.1, "y": -0.05},
    ]
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        initial_population=initial_seeds,
        verbose=False
    )
    
    best, fitness = ga.run()
    print(f"最优解: x={best['x']:.6f}, y={best['y']:.6f}, 适应度: {fitness:.6f}\n")


def example6b():
    """使用自定义初始种群 - 方式2：从点列表生成"""
    from genetic_algorithm import create_seeds_from_points
    
    print("=" * 60)
    print("示例6b: 从点列表生成初始种群")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    # 方式2：从点列表生成（更简洁，适合大量种子）
    points = [
        [0.1, 0.1],
        [-0.2, 0.15],
        [0.05, -0.1],
        [-0.1, -0.05],
        [0.2, -0.1],
        [-0.15, 0.2],
    ]
    initial_seeds = create_seeds_from_points(["x", "y"], points)
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        initial_population=initial_seeds,
        verbose=False
    )
    
    best, fitness = ga.run()
    print(f"最优解: x={best['x']:.6f}, y={best['y']:.6f}, 适应度: {fitness:.6f}\n")


def example6c():
    """使用自定义初始种群 - 方式3：从numpy数组生成"""
    import numpy as np
    from genetic_algorithm import create_seeds_from_array
    
    print("=" * 60)
    print("示例6c: 从numpy数组生成初始种群")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    # 方式3：从numpy数组生成（适合从数值计算中获得的结果）
    seeds_array = np.array([
        [0.1, 0.1],
        [-0.2, 0.15],
        [0.05, -0.1],
        [-0.1, -0.05],
    ])
    initial_seeds = create_seeds_from_array(["x", "y"], seeds_array)
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        initial_population=initial_seeds,
        verbose=False
    )
    
    best, fitness = ga.run()
    print(f"最优解: x={best['x']:.6f}, y={best['y']:.6f}, 适应度: {fitness:.6f}\n")


def example6d():
    """使用自定义初始种群 - 方式4：在指定点周围生成随机种子"""
    from genetic_algorithm import create_seeds_around_points
    
    print("=" * 60)
    print("示例6d: 在指定点周围生成随机种子")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    # 方式4：在已知的好点周围生成随机种子（适合局部搜索）
    centers = [
        {"x": 0.1, "y": 0.1},      # 接近最优解
        {"x": -0.2, "y": 0.15},    # 另一个好点
    ]
    initial_seeds = create_seeds_around_points(
        ["x", "y"],
        centers,
        n_per_center=5,  # 每个中心点生成5个随机种子
        noise_scale=0.05  # 噪声比例
    )
    
    ga = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        initial_population=initial_seeds,
        verbose=False
    )
    
    best, fitness = ga.run()
    print(f"最优解: x={best['x']:.6f}, y={best['y']:.6f}, 适应度: {fitness:.6f}")
    print(f"生成了 {len(initial_seeds)} 个初始种子\n")


if __name__ == "__main__":
    # 运行所有示例
    example1()
    example2()
    example3()
    example4()
    example5()
    example6()   # 方式1：手动定义
    example6b()  # 方式2：从点列表生成
    example6c()  # 方式3：从numpy数组生成
    example6d()  # 方式4：在指定点周围生成
