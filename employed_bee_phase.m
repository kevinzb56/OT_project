function pop = employed_bee_phase(pop, ObjFunc, num_vars, lb, ub, limit)
    sol_per_bee = size(pop, 1);
    for i = 1:sol_per_bee
        
        % Generate a neighbor solution k (k != i) and dimension j
        k = randi(sol_per_bee);
        while k == i
            k = randi(sol_per_bee);
        end
        j = randi(num_vars);
        
        % ABC Search Equation
        phi = 2 * rand() - 1;
        new_sol_pos = pop(i, 1:num_vars);
        new_sol_pos(j) = pop(i, j) + phi * (pop(i, j) - pop(k, j));
        
        % Clamp to bounds
        new_sol_pos = max(new_sol_pos, lb);
        new_sol_pos = min(new_sol_pos, ub);
        
        % Greedy Selection using NSGA-II rules
        
        % Create a temporary population for comparison: [Old Solution; New Solution]
        temp_pop = [pop(i, 1:num_vars); new_sol_pos];
        temp_obj = evaluate_objectives(temp_pop, ObjFunc);
        [temp_fronts, temp_cd] = NonDominatedSort(temp_obj);
        
        % The NonDominatedSort result is for a population of size 2.
        % temp_fronts{1} contains indices of the non-dominated solution(s).
        
        % Rule: The solution is better if it is in the first front (index 1 or 2).
        % If only the new solution is in F1 (i.e., new solution dominates old)
        if temp_fronts{1}(1) == 2 
             pop(i, 1:num_vars) = new_sol_pos;
             pop(i, num_vars + 3) = 0; % Reset trial count
        elseif length(temp_fronts{1}) == 2 % Both are non-dominated by each other
             % Choose based on crowding distance (higher distance is better)
             if temp_cd(2) > temp_cd(1) % New solution has higher crowding distance
                 pop(i, 1:num_vars) = new_sol_pos;
                 pop(i, num_vars + 3) = 0; % Reset trial count
             else
                 pop(i, num_vars + 3) = pop(i, num_vars + 3) + 1; % Increment trial count
             end
        else % Old solution is in F1 (i.e., old solution dominates new)
            pop(i, num_vars + 3) = pop(i, num_vars + 3) + 1; % Increment trial count
        end
    end
end