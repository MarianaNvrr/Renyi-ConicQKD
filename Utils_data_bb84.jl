function optimal_pK(N::Real,dB::Integer, estimator::Bool)

    if f != 1.16
        @warn ("Only f = 1.16 is currently supported for the optimal pK")
    end

    #Optimized values for 
    if N == 1e9
        if estimator             
            optimal_pK = [0.982, 0.979, 0.977, 0.974, 0.97, 0.967, 
                                 0.962, 0.957, 0.952, 0.945, 0.937, 
                                 0.929, 0.92, 0.91, 0.896, 0.882, 
                                 0.864, 0.845, 0.821, 0.799, 0.772, 
                                 0.748, 0.705, 0.672, 0.624, 0.588] #48
        else
            optimal_pK = [0.905, 0.89, 0.87, 0.855, 0.83, 0.805, #10
                                 0.775, 0.74, 0.7, 0.66, 0.605, #20
                                 0.55, 0.485, 0.405, 0.315, 0.215] #30
        end
    elseif N == 1e7
        if estimator  
            optimal_pK = [0.937, 0.929, 0.919, 0.908, 0.895, 0.881, 
                                 0.864, 0.846, 0.825, 0.801, 0.773, 
                                 0.744, 0.71, 0.675, 0.63, 0.59]  #30
        else
            optimal_pK = [0.802, 0.772, 0.736, 0.698, 0.652, 0.602, 
                                 0.544, 0.476, 0.4, 0.31, 0.196]
        end
    elseif N == 1e5
        if estimator  
            optimal_pK = [0.772, 0.742, 0.709, 0.672, 0.632, 0.592]
        else
            #only valid for dB = vcat(0:2:6,7)
            optimal_pK = [0.589, 0.53, 0.462, 0.383, 0.337] 
        end
    elseif N==1e6
        if estimator==false
            optimal_pK = [0.714, 0.672, 0.624, 0.569, 0.507, 0.434, 
                             0.35, 0.245]
        end
    else
        @warn("N given is not supported for the optimal pK")
    end
    return optimal_pK[Int(ceil(dB/2))+1] 
end
