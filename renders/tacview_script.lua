-- TacView Lua script for combat visualization
-- Controls camera behavior and zoom levels

function OnLoad()
    -- Set initial camera settings
    TacView.SetCameraMode(0)  -- 0 = Follow mode
    TacView.SetCameraDistance(5000)  -- Initial distance in meters
    TacView.SetCameraFOV(60)  -- Field of view in degrees
end

function OnFrame()
    -- Get the current time
    local time = TacView.GetTime()
    
    -- Adjust camera distance based on combat situation
    local distance = TacView.GetCameraDistance()
    local targetDistance = 5000  -- Default distance
    
    -- Get all objects in view
    local objects = TacView.GetObjects()
    if objects then
        -- Find the closest distance between any two objects
        local minDistance = 100000
        for i, obj1 in ipairs(objects) do
            for j, obj2 in ipairs(objects) do
                if i ~= j then
                    local dist = TacView.GetDistance(obj1, obj2)
                    if dist < minDistance then
                        minDistance = dist
                    end
                end
            end
        end
        
        -- Adjust camera distance based on object separation
        if minDistance < 100000 then
            targetDistance = math.max(2000, minDistance * 2)
        end
    end
    
    -- Smoothly adjust camera distance
    if math.abs(distance - targetDistance) > 100 then
        TacView.SetCameraDistance(distance + (targetDistance - distance) * 0.1)
    end
end

function OnKeyDown(key)
    -- Add keyboard controls if needed
    if key == 0x5A then  -- 'Z' key
        TacView.SetCameraDistance(5000)  -- Reset zoom
    end
end 