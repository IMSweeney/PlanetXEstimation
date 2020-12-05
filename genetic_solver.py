import logging
import random
import tqdm


class GeneticSolver():
    def __init__(self, x_i, score_fcn, args=[], kwargs={},
                 num_iterations=1000, pop_size=1000,
                 selection=0.1, mut_chance=0.1, seed=None):
        self.x_i = x_i
        self.score_fcn = score_fcn
        self.fcn_args = args
        self.fcn_kwargs = kwargs
        self.pop_size = pop_size
        self.selection_ = selection
        self.crossover_ = len(x_i) // 2
        self.mutation_ = {'chance': mut_chance, 'amount_max': 0.05,
                          'min': 1e-4}

        self.pop = None
        self.num_iterations = num_iterations

        self.rand = random.Random(seed)
        self.logger = logging.getLogger(self.__class__.__name__)

    def solve(self):
        self.generate_population()
        # Might be better to use random.gauss(mu, sigma) instead?
        for i in range(3):  # Mutate the initial distribution more
            self.mutation()
        self.score_population()

        for i in tqdm.tqdm(range(self.num_iterations)):
            self.selection()
            self.crossover()
            self.mutation()
            self.score_population()
        return self.get_best_individual()

    def generate_population(self):

        self.pop = []
        for i in range(self.pop_size):
            self.pop.append(Individual(self.x_i))

    def score_population(self):
        for i in self.pop:
            i.score = self.score_fcn(i.x, *self.fcn_args, **self.fcn_kwargs)

    def selection(self):
        self.pop = sorted(self.pop, key=lambda i: i.score)
        selection_cutoff = int(self.pop_size * self.selection_)
        self.pop = self.pop[:selection_cutoff]

    def crossover(self):
        num_children_needed = self.pop_size - len(self.pop)
        parent_vecs = [i.x for i in self.pop]
        parent_scores = [i.score for i in self.pop]
        p1 = self.rand.choices(parent_vecs, weights=parent_scores,
                               k=num_children_needed)
        p2 = self.rand.choices(parent_vecs, weights=parent_scores,
                               k=num_children_needed)
        for i in range(num_children_needed):
            p1_gene = p1[i][:self.crossover_]
            p2_gene = p2[i][self.crossover_:]
            child_gene = p1_gene + p2_gene
            self.pop.append(Individual(child_gene))

        assert(len(self.pop) == self.pop_size)

    def mutation(self):
        mut_chance = self.mutation_['chance']
        max_amount = self.mutation_['amount_max']
        min_adjust = self.mutation_['min']

        # seems very inefficient
        for i in self.pop:
            vec = i.x
            new_vec = []
            for val in vec:
                if self.rand.random() < mut_chance:
                    mut_amount = val * max_amount * self.rand.uniform(-1, 1)
                    if mut_amount > 0:
                        mut_amount = max(min_adjust, mut_amount)
                    else:
                        mut_amount = min(-1 * min_adjust, mut_amount)
                    new_vec.append(val + mut_amount)
                else:
                    new_vec.append(val)
            i.x = new_vec

    def get_best_individual(self):
        best = sorted(self.pop, key=lambda i: i.x)[0]
        return best.x, best.score


class Individual():
    def __init__(self, x):
        self.x = x
        self.score = None


def test_score_fcn(x, p, a=None):
    # print(p, a)
    score = 0
    best = [10.1, 0.2, -12.5, 0, -0.00001]
    for i, val in enumerate(x):
        score += (val - best[i]) ** 2
    return score ** (1 / 2)


if __name__ == '__main__':
    x_i = [10, 0, 0, 0, 0]
    gen = GeneticSolver(x_i, test_score_fcn, args=['a'], kwargs={'a': '3'})
    best, score = gen.solve()
    print('Best score: {}'.format(score))
    print('Vec: {}'.format(best))
