"""
遗传算法模板
支持变量范围定义、适应度惩罚和约束条件判断
Python 3.10+ 兼容
"""

import numpy as np
import random
from typing import Callable


# ==================== 辅助函数：生成初始种群 ====================

def create_seeds_from_points(
    variable_names: list[str],
    points: list[list[float] | tuple[float, ...]]
) -> list[dict[str, float]]:
    """
    从点列表创建初始种子
    
    参数:
        variable_names: 变量名列表，例如 ["x", "y"]
        points: 点列表，每个点是一个列表或元组，例如 [[0.1, 0.1], [-0.2, 0.15]]
    
    返回:
        初始种群列表
    
    示例:
        seeds = create_seeds_from_points(
            ["x", "y"],
            [[0.1, 0.1], [-0.2, 0.15], [0.05, -0.1]]
        )
    """
    return [
        {var_name: point[i] for i, var_name in enumerate(variable_names)}
        for point in points
    ]


def create_seeds_from_array(
    variable_names: list[str],
    array: np.ndarray
) -> list[dict[str, float]]:
    """
    从numpy数组创建初始种子
    
    参数:
        variable_names: 变量名列表
        array: numpy数组，形状为 (n_points, n_variables)
    
    返回:
        初始种群列表
    
    示例:
        seeds = create_seeds_from_array(
            ["x", "y"],
            np.array([[0.1, 0.1], [-0.2, 0.15]])
        )
    """
    return [
        {var_name: float(array[i, j]) for j, var_name in enumerate(variable_names)}
        for i in range(array.shape[0])
    ]


def create_seeds_around_points(
    variable_names: list[str],
    centers: list[dict[str, float]],
    n_per_center: int = 1,
    noise_scale: float = 0.1
) -> list[dict[str, float]]:
    """
    在指定点周围生成随机种子
    
    参数:
        variable_names: 变量名列表
        centers: 中心点列表，每个中心点是一个字典
        n_per_center: 每个中心点周围生成的种子数量
        noise_scale: 噪声比例（相对于变量范围）
    
    返回:
        初始种群列表
    
    示例:
        seeds = create_seeds_around_points(
            ["x", "y"],
            [{"x": 0.1, "y": 0.1}, {"x": -0.2, "y": 0.15}],
            n_per_center=5,
            noise_scale=0.05
        )
    """
    seeds = []
    for center in centers:
        for _ in range(n_per_center):
            seed = {}
            for var_name in variable_names:
                base_value = center.get(var_name, 0.0)
                # 简单噪声：在实际应用中可能需要知道变量范围
                seed[var_name] = base_value + random.gauss(0, noise_scale)
            seeds.append(seed)
    return seeds


def load_seeds_from_file(filepath: str) -> list[dict[str, float]]:
    """
    从JSON文件加载初始种子（需要json模块）
    
    参数:
        filepath: JSON文件路径，文件格式应为列表的列表或字典列表
    
    返回:
        初始种群列表
    
    示例文件格式 (seeds.json):
        [
            {"x": 0.1, "y": 0.1},
            {"x": -0.2, "y": 0.15}
        ]
    """
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            return data
        else:
            raise ValueError("JSON文件格式错误：应为字典列表")
    else:
        raise ValueError("JSON文件格式错误：应为列表")


class GeneticAlgorithm:
    """遗传算法类"""
    
    def __init__(
        self,
        variable_ranges: dict[str, tuple[float, float]],
        fitness_function: Callable[[dict[str, float]], float],
        population_size: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        max_generations: int = 100,
        penalty_function: Callable[[dict[str, float], float], float] | None = None,
        constraint_function: Callable[[dict[str, float]], bool] | None = None,
        elitism_count: int = 2,
        verbose: bool = True,
        initial_population: list[dict[str, float]] | None = None,
        maximize: bool = False
    ):
        """
        初始化遗传算法
        
        参数:
            variable_ranges: 变量范围字典，例如 {"x": (-1, 1), "y": (0, 10)}
            fitness_function: 适应度函数，输入变量字典，返回适应度值
            population_size: 种群大小
            crossover_rate: 交叉率
            mutation_rate: 变异率
            max_generations: 最大迭代代数
            penalty_function: 惩罚函数，输入(变量字典, 原始适应度)，返回惩罚后的适应度
            constraint_function: 约束函数，输入变量字典，返回True表示满足约束，False表示不满足
            elitism_count: 精英保留数量
            verbose: 是否打印过程信息
            initial_population: 初始种群（种子），如果提供则使用，否则随机生成
            maximize: 是否为最大化问题，True表示最大化，False表示最小化（默认）
        """
        self.variable_ranges = variable_ranges
        self.variable_names = list(variable_ranges.keys())
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.penalty_function = penalty_function
        self.constraint_function = constraint_function
        self.elitism_count = elitism_count
        self.verbose = verbose
        self.initial_population = initial_population
        self.maximize = maximize
        
        # 存储每代最佳适应度
        self.best_fitness_history = []
        self.best_individual_history = []
        
    def _generate_random_individual(self) -> dict[str, float]:
        """生成一个随机个体"""
        individual = {}
        for var_name, (min_val, max_val) in self.variable_ranges.items():
            individual[var_name] = random.uniform(min_val, max_val)
        return individual
    
    def _validate_individual(self, individual: dict[str, float]) -> bool:
        """验证个体是否有效（所有变量都在范围内）"""
        for var_name in self.variable_names:
            if var_name not in individual:
                return False
            min_val, max_val = self.variable_ranges[var_name]
            if not (min_val <= individual[var_name] <= max_val):
                return False
        return True
    
    def _initialize_population(self) -> list[dict[str, float]]:
        """初始化种群"""
        if self.initial_population is not None:
            # 使用用户提供的初始种群
            population = []
            for ind in self.initial_population:
                # 验证并复制个体
                if self._validate_individual(ind):
                    population.append(ind.copy())
                else:
                    # 如果个体无效，生成随机个体替代
                    if self.verbose:
                        print("警告: 初始种群中的某些个体无效，已用随机个体替代")
                    population.append(self._generate_random_individual())
            
            # 如果初始种群数量不足，用随机个体填充
            while len(population) < self.population_size:
                population.append(self._generate_random_individual())
            
            # 如果初始种群数量过多，截断到指定大小
            population = population[:self.population_size]
            
            if self.verbose and self.initial_population:
                print(f"使用自定义初始种群 ({len(self.initial_population)} 个个体)")
        else:
            # 随机生成种群
            population = []
            for _ in range(self.population_size):
                population.append(self._generate_random_individual())
        
        return population
    
    def _evaluate_fitness(self, individual: dict[str, float]) -> float:
        """评估个体适应度（包含约束检查和惩罚）"""
        # 检查约束条件
        if self.constraint_function is not None:
            if not self.constraint_function(individual):
                # 约束不满足，返回惩罚值
                if self.maximize:
                    return -1e10  # 最大化问题，返回非常小的值
                else:
                    return 1e10   # 最小化问题，返回非常大的值
        
        # 计算原始适应度
        raw_fitness = self.fitness_function(individual)
        
        # 应用惩罚函数
        if self.penalty_function is not None:
            fitness = self.penalty_function(individual, raw_fitness)
        else:
            fitness = raw_fitness
        
        return fitness
    
    def _select_parents(self, population: list[dict[str, float]], 
                       fitnesses: list[float]) -> tuple[dict[str, float], dict[str, float]]:
        """选择父代（轮盘赌选择）"""
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        # 避免除零
        if max_fitness == min_fitness:
            return random.choice(population), random.choice(population)
        
        # 根据问题类型处理适应度值
        if self.maximize:
            # 最大化问题：适应度值越大越好，直接使用
            # 转换为非负值用于选择
            selection_fitnesses = [f - min_fitness + (max_fitness - min_fitness) * 0.1 
                                  for f in fitnesses]
        else:
            # 最小化问题：适应度值越小越好，转换为选择概率
            # 将最小化问题转换为最大化问题（用于选择）
            selection_fitnesses = [max_fitness - f + (max_fitness - min_fitness) * 0.1 
                                  for f in fitnesses]
        total_fitness = sum(selection_fitnesses)
        
        # 选择第一个父代
        rand1 = random.uniform(0, total_fitness)
        cumsum = 0
        parent1 = population[0]
        for i, sel_fit in enumerate(selection_fitnesses):
            cumsum += sel_fit
            if cumsum >= rand1:
                parent1 = population[i]
                break
        
        # 选择第二个父代
        rand2 = random.uniform(0, total_fitness)
        cumsum = 0
        parent2 = population[0]
        for i, sel_fit in enumerate(selection_fitnesses):
            cumsum += sel_fit
            if cumsum >= rand2:
                parent2 = population[i]
                break
        
        return parent1, parent2
    
    def _crossover(self, parent1: dict[str, float], parent2: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """交叉操作（算术交叉）"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = {}
        child2 = {}
        
        alpha = random.uniform(0, 1)
        
        for var_name in self.variable_names:
            val1 = parent1[var_name]
            val2 = parent2[var_name]
            child1[var_name] = alpha * val1 + (1 - alpha) * val2
            child2[var_name] = (1 - alpha) * val1 + alpha * val2
            
            # 确保在范围内
            min_val, max_val = self.variable_ranges[var_name]
            child1[var_name] = np.clip(child1[var_name], min_val, max_val)
            child2[var_name] = np.clip(child2[var_name], min_val, max_val)
        
        return child1, child2
    
    def _mutate(self, individual: dict[str, float], generation: int) -> dict[str, float]:
        """变异操作（非均匀变异）"""
        mutated = individual.copy()
        
        for var_name in self.variable_names:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.variable_ranges[var_name]
                # 非均匀变异，随着代数增加，变异幅度减小
                mutation_range = (max_val - min_val) * (1 - generation / self.max_generations) * 0.1
                mutation = random.gauss(0, mutation_range)
                mutated[var_name] = np.clip(mutated[var_name] + mutation, min_val, max_val)
        
        return mutated
    
    def run(self) -> tuple[dict[str, float], float]:
        """
        运行遗传算法
        
        返回:
            (最优个体, 最优适应度值)
        """
        # 初始化种群
        population = self._initialize_population()
        
        # 评估初始种群
        fitnesses = [self._evaluate_fitness(ind) for ind in population]
        
        # 记录最佳个体
        if self.maximize:
            best_idx = np.argmax(fitnesses)  # 最大化问题
        else:
            best_idx = np.argmin(fitnesses)  # 最小化问题
        best_individual = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        self.best_fitness_history.append(best_fitness)
        self.best_individual_history.append(best_individual.copy())
        
        if self.verbose:
            print(f"初始最佳适应度: {best_fitness:.6f}")
        
        # 迭代进化
        for generation in range(1, self.max_generations + 1):
            new_population = []
            
            # 精英保留
            if self.maximize:
                sorted_indices = np.argsort(fitnesses)[::-1]  # 降序排列（最大化问题）
            else:
                sorted_indices = np.argsort(fitnesses)  # 升序排列（最小化问题）
            for i in range(self.elitism_count):
                new_population.append(population[sorted_indices[i]].copy())
            
            # 生成新种群
            while len(new_population) < self.population_size:
                # 选择
                parent1, parent2 = self._select_parents(population, fitnesses)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1, generation)
                child2 = self._mutate(child2, generation)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小正确
            population = new_population[:self.population_size]
            
            # 评估新种群
            fitnesses = [self._evaluate_fitness(ind) for ind in population]
            
            # 更新最佳个体
            if self.maximize:
                current_best_idx = np.argmax(fitnesses)
            else:
                current_best_idx = np.argmin(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if self.maximize:
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[current_best_idx].copy()
            else:
                if current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[current_best_idx].copy()
            
            self.best_fitness_history.append(best_fitness)
            self.best_individual_history.append(best_individual.copy())
            
            # 输出进度信息：每10代输出一次，或者如果总代数少于50代则每代都输出
            if self.verbose:
                if self.max_generations <= 50:
                    # 如果总代数较少，每代都输出
                    print(f"第 {generation} 代 - 最佳适应度: {best_fitness:.6f}")
                elif generation % 10 == 0:
                    # 否则每10代输出一次
                    print(f"第 {generation} 代 - 最佳适应度: {best_fitness:.6f}")
                elif generation % 5 == 0:
                    # 每5代输出一个简单的进度提示（使用\r覆盖当前行）
                    print(f"第 {generation} 代...", end="\r", flush=True)
        
        if self.verbose:
            print("\n优化完成！")
            print(f"最优适应度: {best_fitness:.6f}")
            print(f"最优个体: {best_individual}")
        
        return best_individual, best_fitness


# 使用示例
if __name__ == "__main__":
    # 示例1: 简单的二次函数优化（无约束）
    print("=" * 50)
    print("示例1: 最小化 f(x, y) = x^2 + y^2")
    print("=" * 50)
    
    def fitness_func1(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x**2 + y**2
    
    ga1 = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_func1,
        population_size=50,
        max_generations=100
    )
    
    best1, fitness1 = ga1.run()
    print(f"\n结果: x={best1['x']:.6f}, y={best1['y']:.6f}, f={fitness1:.6f}\n")
    
    # 示例2: 带约束条件的优化
    print("=" * 50)
    print("示例2: 带约束条件 x + y >= 0.5")
    print("=" * 50)
    
    def constraint_func2(vars_dict):
        x = vars_dict["x"]
        y = vars_dict["y"]
        return x + y >= 0.5  # 约束条件
    
    ga2 = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_func1,
        constraint_function=constraint_func2,
        population_size=50,
        max_generations=100
    )
    
    best2, fitness2 = ga2.run()
    print(f"\n结果: x={best2['x']:.6f}, y={best2['y']:.6f}, f={fitness2:.6f}")
    print(f"约束检查: x + y = {best2['x'] + best2['y']:.6f} >= 0.5\n")
    
    # 示例3: 使用惩罚函数
    print("=" * 50)
    print("示例3: 使用惩罚函数处理约束")
    print("=" * 50)
    
    def penalty_func3(vars_dict, raw_fitness):
        x = vars_dict["x"]
        y = vars_dict["y"]
        # 软约束：x + y >= 0.5，不满足时添加惩罚
        penalty = 0
        if x + y < 0.5:
            penalty = 100 * (0.5 - (x + y))**2
        return raw_fitness + penalty
    
    ga3 = GeneticAlgorithm(
        variable_ranges={"x": (-1, 1), "y": (-1, 1)},
        fitness_function=fitness_func1,
        penalty_function=penalty_func3,
        population_size=50,
        max_generations=100
    )
    
    best3, fitness3 = ga3.run()
    print(f"\n结果: x={best3['x']:.6f}, y={best3['y']:.6f}, f={fitness3:.6f}")
    print(f"约束检查: x + y = {best3['x'] + best3['y']:.6f} >= 0.5\n")
