class Evaluations:
    # Method to calculate Equal Error Rate (EER)
    def calculate_eer(self, false_acceptance_rates, false_rejection_rates):
        eer = 0.0
        for far, frr in zip(false_acceptance_rates, false_rejection_rates):
            if far == frr:
                eer = far
                break
        return eer

    # Method to calculate False Acceptance Rate (FAR)
    def calculate_far(self, false_acceptances, total_impostor_attempts):
        if total_impostor_attempts == 0:
            return 0.0
        return false_acceptances / total_impostor_attempts

    # Method to calculate False Rejection Rate (FRR)
    def calculate_frr(self, false_rejections, total_genuine_attempts):
        if total_genuine_attempts == 0:
            return 0.0
        return false_rejections / total_genuine_attempts
