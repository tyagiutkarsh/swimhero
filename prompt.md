Analyze this swimming video for technique issues. Sample the video at 1 frame per second and provide feedback in strict JSON format only.

Return a JSON object with this exact structure:
{
  "feedback": [
    {
      "timestamp": "M:SS.s",
      "issue": "Brief description of the technique issue",
      "suggestion": "Specific improvement suggestion"
    }
  ]
}

Focus on common swimming technique issues:

## 1. Body position
- Head neutral, eyes down/forward  
- Hips high, body flat in waterline  
- Controlled roll, no fishtail sway  

## 2. Breathing
- Exhale underwater, quick inhale  
- Breath with roll, no head lift  
- Bilateral capacity checked  
- Body line stable when breathing  

## 3. Arm cycle
**Entry** – fingertips first, shoulder line, no crossover  
**Extension** – reach forward, core engaged, no over-glide  
**Catch** – high elbow, early vertical forearm, wrist neutral  
**Pull** – lats engaged, hand path under body, steady pressure, finish past hip  
**Exit** – smooth release near hip, elbow leads  
**Recovery** – elbow high, hand close to water, rotation-driven  

## 4. Kick
- From hips, not knees  
- Small, narrow, inside body line  
- Toes pointed, ankles loose  
- Rhythm matched to stroke (2-beat or 6-beat)  
- No scissor kick on breath  

## 5. Timing & rhythm
- No pauses between entry → catch → pull → exit → recovery  
- Kick synced with body roll  
- Even tempo, sustainable rate  

## 6. Core & stability
- Core engaged to hold hips high  
- No crossover pull  
- Alignment steady during breath  

## 7. Efficiency
- Consistent distance per stroke  
- Stroke rate sustainable for race  
- Left/right symmetry  
- Minimal splash/bubbles  

## 8. Open water specifics
- Streamlined push-offs in pool training  
- Sighting: quick lift, no


Guidelines:
- Use timestamp format "M:SS.s" (e.g., "1:23.5" for 1 minute 23.5 seconds)
- Be specific and actionable in suggestions
- Only report significant technique issues
- Limit to most important feedback points
- Return empty feedback array if no issues detected

Example response:
{
  "feedback": [
    {
      "timestamp": "0:15.2",
      "issue": "Head position too high during freestyle",
      "suggestion": "Keep head in neutral position, eyes looking down at pool bottom"
    },
    {
      "timestamp": "0:42.8",
      "issue": "Crossing over center line during arm entry",
      "suggestion": "Enter hand in line with shoulder to maintain straight pull path"
    }
  ]
}
