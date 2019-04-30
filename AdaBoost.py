import math

class Adboost(object):
    def __init__(self, x_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       y_list = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]):
        """initialization
        initializing parameters of the Adboost

        input
        x_list : training data set
        y_list : labels of the x_list
        """
        self.x_list = x_list
        self.y_list = y_list
        self.sample_count = len(self.y_list)
        self.dt_list = [0.1] * self.sample_count  #the distribution of sample weights in the current round
        self.dt_next_list = []  #the distribution of sample weights in the next round
        self.ht_list = []  #results of the base classifier in the current round
        self.alpha = 0.0  #the weight of the self.ht_list
        self.Ht_list = [0.0] * self.sample_count  #results of the Adboost  in the current round
        self.Ht_sign_list = []  #sign(self.Ht_list)
        self.acc = 0.0  #the accuracy rate of the Adboost in the current round

    def __get_decision_stump(self):
        """base classifier(decision stump)
        obtain the decision stump by minimizing the classification error of current round with the self.dt_list

        output
        error_list  :  the error of each theta value and method
        method_flag :  method flag
        error_min   :  the minimum error of the error_list
        theta_min   :  the value of theta of the corresponding error_min

        error_min is necessary for return to compute other parameters, the rest are not
        """

        def __get_ht_list(theta):
            ht_list = []
            for x in self.x_list:
                if x < theta:
                    ht_list.append(1)
                else:
                    ht_list.append(-1)
            return ht_list

        theta_list = [x + 0.5 for x in self.x_list[:-1]]
        error_list = [[], []]
        error_min = 99999
        for theta in theta_list:
            ht_list_1 = __get_ht_list(theta) #method 1
            ht_list_2 = [-ht for ht in ht_list_1] #method 2
            for method, ht_list in enumerate([ht_list_1, ht_list_2]):
                error = 0
                for loc, y_ in enumerate(ht_list):
                    if y_ != self.y_list[loc]:
                        error += self.dt_list[loc]
                error_list[method].append(error)
                if error < error_min:
                    error_min = error
                    theta_min = theta
                    method_flag = method + 1
                    self.ht_list = ht_list
        return error_list, method_flag, error_min, theta_min 

    def __get_alpha(self, error):
        self.alpha = 0.5 * math.log((1-error)/(error))
        
    def __get_dt_next(self):
        for loc, dt in enumerate(self.dt_list):
            z = math.e**(-self.alpha * self.y_list[loc] * self.ht_list[loc])
            self.dt_next_list.append(dt * z)
        self.dt_next_list = [dt / sum(self.dt_next_list) for dt in self.dt_next_list]
        
    def __get_acc_rate(self):
        error = 0
        for loc, _ in enumerate(self.x_list):
            self.Ht_list[loc] += self.alpha * self.ht_list[loc]
        self.Ht_sign_list = [int(math.pow(-1, (Ht > 0) * 1 + 1)) for Ht in self.Ht_list]
        self.acc = map(cmp, self.Ht_sign_list, self.y_list).count(0) / (self.sample_count + 0.0)

    def train(self, T = 3):
        for t in range(T):
            error_list, method, error, theta = self.__get_decision_stump()
            self.__get_alpha(error)
            self.__get_dt_next()
            self.__get_acc_rate()

            print '--------------Round %s--------------' % (t+1)
            print 'error list : %s\n\t\t\t %s' % ([round(e, 3) for e in error_list[0]],\
                                                  [round(e, 3) for e in error_list[1]])
            print 'theta : %s' % theta
            print 'method : %s' % method
            print 'error : %s' % round(error, 4)
            print 'alpha : %s' % round(self.alpha, 4)
            print 'ht_list : %s' % self.ht_list
            print 'dt_list : %s' % [round(dt, 3) for dt in self.dt_list]
            print 'dt_next_list : %s' % [round(dt, 3) for dt in self.dt_next_list]
            print 'Ht_list : %s' % [round(Ht, 3) for Ht in self.Ht_list]
            print 'Ht_sign_list : %s' % self.Ht_sign_list
            print 'accuracy rate : %s' % self.acc
            print '------------------------------------\n'
            self.dt_list = self.dt_next_list
            self.dt_next_list = []

            if self.acc == 1.0:
                break

if __name__ == "__main__":
    adb = Adboost()
    adb.train(5)