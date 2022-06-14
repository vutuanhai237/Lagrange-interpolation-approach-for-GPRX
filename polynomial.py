class Polymonial():
    def __init__(self, coeffs: list):
        self.coeff = coeffs
        self.deg = len(coeffs)

    def __str__(self):
        poly = str(self.coeff[0])
        for i in range(1, self.deg):
            poly += ' + ' + str(self.coeff[i]) + 'G^' + str(i)
        return poly

    def multi(self, another_poly):
        result = Polymonial([0]*(self.deg + another_poly.deg - 1))
        for i in range(0, self.deg):
            for j in range(0, another_poly.deg):
                result.coeff[i+j] += self.coeff[i] * another_poly.coeff[j]
        return result
    def add(self, another_poly):
        
        if self.deg > another_poly.deg:
            result = Polymonial([0]*self.deg)
            lower_deg = another_poly.deg
            for i in range(lower_deg, self.deg):
                result.coeff[i] = self.coeff[i]
        else:
            result = Polymonial([0]*another_poly.deg)
            lower_deg = self.deg
            for i in range(lower_deg, another_poly.deg):
                result.coeff[i] = another_poly.coeff[i]
        for i in range(0, lower_deg):
            result.coeff[i] += self.coeff[i] + another_poly.coeff[i]
        
        return result
    def multiX(self, scalar):
        for i in range(0, self.deg):
            self.coeff[i] *= scalar
        return self

def multiXPoly(polys):
    results = polys[0]
    for i in range(1, len(polys)):
        results = results.multi(polys[i])

    return results

def addXPoly(polys):
    results = polys[0]
    for i in range(1, len(polys)):
        results = results.add(polys[i])
    return results